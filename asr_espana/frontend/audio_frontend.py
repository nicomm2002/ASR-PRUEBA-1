"""
Fase 1 — Frontend de audio para ASR en español de España.

Pipeline:
  1. Carga y resampling a 16 kHz mono
  2. Normalización de amplitud
  3. VAD (patrones prosódicos peninsulares)
  4. Pre-emphasis filter
  5. Framing: ventanas 25 ms, salto 10 ms
  6. Ventana Hann
  7. STFT / FFT  (N=512)
  8. Banco de filtros Mel  (80 filtros)
  9. Log-compresión
 10. CMVN por utterance
 11. Salida: tensor [batch, T, 80]
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    import torchaudio.functional as TAF
    import torchaudio.transforms as TAT
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

from ..config import AudioConfig
from .vad import SpanishVAD


class AudioFrontend(nn.Module):
    """
    Frontend completo de audio.

    Args:
        config: Instancia de ``AudioConfig`` con todos los hiperparámetros.

    Entrada:
        waveform : Tensor ``[batch, samples]`` o ``[batch, 1, samples]``
                   en la frecuencia ``src_sample_rate``.
    Salida:
        features : Tensor ``[batch, T, n_mels]``
        lengths  : Tensor ``[batch]`` con las longitudes reales de cada
                   secuencia (en frames), tras VAD y subsampling.
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.cfg = config

        # --- Banco de filtros Mel (buffer, no parámetro entrenable) ---
        self.register_buffer(
            "mel_filterbank",
            self._build_mel_filterbank(),
        )

        # --- VAD peninsular ---
        self.vad = SpanishVAD(config)

    # ------------------------------------------------------------------
    # Interfaz principal
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        src_sample_rate: int = 16_000,
        lengths: Optional[torch.Tensor] = None,
        apply_vad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa una forma de onda y devuelve features Mel + longitudes.

        Args:
            waveform       : ``[B, T_audio]`` o ``[B, 1, T_audio]`` (float32).
            src_sample_rate: Frecuencia de muestreo de la señal de entrada.
            lengths        : ``[B]`` longitudes en muestras (opcional).
            apply_vad      : Si True aplica VAD antes de extraer features.

        Returns:
            features : ``[B, T_frames, 80]``
            lengths  : ``[B]`` longitudes en frames.
        """
        # 1. Asegurar forma [B, T]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        batch_size = waveform.size(0)

        # 2. Resampling → 16 kHz
        if src_sample_rate != self.cfg.sample_rate:
            waveform = self._resample(waveform, src_sample_rate, self.cfg.sample_rate)

        # 3. Normalización de amplitud (por muestra del batch)
        waveform = self._normalize_amplitude(waveform)

        # 4. VAD (opcional; útil en inferencia)
        if apply_vad:
            waveform, lengths = self.vad(waveform, lengths)

        # 5. Calcular longitudes en frames si no se proporcionaron
        if lengths is None:
            lengths = torch.full(
                (batch_size,),
                waveform.size(1),
                dtype=torch.long,
                device=waveform.device,
            )

        # 6. Pre-énfasis
        waveform = self._pre_emphasis(waveform)

        # 7-9. STFT → Mel → log
        features = self._extract_log_mel(waveform)  # [B, T, 80]

        # 10. CMVN por utterance
        if self.cfg.cmvn:
            features = self._cmvn(features)

        # 11. Calcular longitudes en frames
        frame_lengths = self._samples_to_frames(lengths)

        return features, frame_lengths

    # ------------------------------------------------------------------
    # Pasos del pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
    ) -> torch.Tensor:
        """Resampling usando torchaudio si está disponible, sino lineal."""
        if _TORCHAUDIO_AVAILABLE:
            return TAF.resample(waveform, orig_freq=orig_freq, new_freq=new_freq)
        # Fallback: interpolación lineal (no ideal, solo para compatibilidad)
        ratio = new_freq / orig_freq
        new_len = int(waveform.size(-1) * ratio)
        return F.interpolate(
            waveform.unsqueeze(1), size=new_len, mode="linear", align_corners=False
        ).squeeze(1)

    @staticmethod
    def _normalize_amplitude(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalización por utterance: divide entre el máximo absoluto."""
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
        return waveform / peak

    def _pre_emphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Filtro de pre-énfasis:  y[t] = x[t] − α·x[t−1]
        Realzar frecuencias altas compensa la caída natural del espectro vocal.
        """
        alpha = self.cfg.pre_emphasis_coeff
        # Shifteamos y restamos con coeficiente
        emphasized = waveform.clone()
        emphasized[:, 1:] = waveform[:, 1:] - alpha * waveform[:, :-1]
        return emphasized

    def _extract_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extrae features log-Mel del espectrograma.

        Pipeline:
          STFT  →  |·|²  →  banco Mel  →  log(·)
        """
        cfg = self.cfg
        win = torch.hann_window(cfg.win_length, device=waveform.device)

        # STFT: [B, n_fft/2+1, T]
        stft = torch.stft(
            waveform,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=win,
            return_complex=True,
            pad_mode="reflect",
            center=True,
        )

        # Espectrograma de potencia: [B, F, T]
        power_spec = stft.abs().pow(2)

        # Banco de filtros Mel: [B, n_mels, T]
        # mel_filterbank: [n_mels, n_fft/2+1]
        mel_spec = torch.matmul(self.mel_filterbank, power_spec)

        # Log-compresión
        log_mel = torch.log(mel_spec.clamp(min=cfg.log_floor))

        # Transponer → [B, T, n_mels]
        return log_mel.transpose(1, 2)

    @staticmethod
    def _cmvn(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Cepstral Mean and Variance Normalization por utterance.
        Resta la media y divide entre la desviación estándar por canal.
        """
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True).clamp(min=eps)
        return (features - mean) / std

    def _samples_to_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Convierte longitudes en muestras a longitudes en frames.
        Con center=True, torch.stft añade padding de win_length//2 a cada lado,
        por lo que: n_frames = ceil(n_samples / hop_length).
        """
        cfg = self.cfg
        frame_lengths = torch.div(
            lengths + cfg.hop_length - 1, cfg.hop_length, rounding_mode="floor"
        )
        return frame_lengths.long()

    # ------------------------------------------------------------------
    # Banco de filtros Mel
    # ------------------------------------------------------------------

    def _build_mel_filterbank(self) -> torch.Tensor:
        """
        Construye el banco de filtros Mel triangulares.
        Devuelve tensor ``[n_mels, n_fft//2 + 1]``.
        """
        cfg = self.cfg
        n_freqs = cfg.n_fft // 2 + 1

        mel_min = self._hz_to_mel(torch.tensor(cfg.f_min)).item()
        mel_max = self._hz_to_mel(torch.tensor(cfg.f_max)).item()
        mel_points = torch.linspace(mel_min, mel_max, cfg.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Índices en el eje de frecuencia
        bin_points = torch.floor(
            (cfg.n_fft + 1) * hz_points / cfg.sample_rate
        ).long()

        filterbank = torch.zeros(cfg.n_mels, n_freqs)
        for m in range(1, cfg.n_mels + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]

            # Rampa ascendente
            for k in range(f_left, f_center):
                if f_center - f_left > 0:
                    filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            # Rampa descendente
            for k in range(f_center, f_right):
                if f_right - f_center > 0:
                    filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return filterbank  # [n_mels, n_fft//2+1]

    # ------------------------------------------------------------------
    # Helpers escala Mel
    # ------------------------------------------------------------------

    @staticmethod
    def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + freq / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
