"""
VAD (Voice Activity Detection) ajustado a los patrones prosódicos
del español peninsular.

Características del español de España relevantes para el VAD:
  - Ritmo silábico (syllable-timed): sílabas de duración más uniforme
    que en inglés (stress-timed).
  - Velocidad elocutiva media ~5.7 sílabas/s en habla espontánea.
  - Pausas inter-frase cortas (~200-400 ms), sin silencios largos
    entre palabras.
  - Mayor energía en vocales /a/, /e/ respecto a consonantes.
  - Nasales /m/, /n/, /ñ/ con energía espectral concentrada en bajas
    frecuencias (< 1 kHz).

El algoritmo usa una combinación de:
  1. Energía de corta duración (STE)
  2. Cruce por cero (ZCR) — para sibilantes /s/, /z/, /c/
  3. Suavizado temporal con mínima duración de segmento de voz
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SpanishVAD(nn.Module):
    """
    Detector de actividad de voz optimizado para español peninsular.

    Combina energía de corta duración (STE) y tasa de cruce por cero
    (ZCR) con umbrales calibrados para el ritmo silábico del castellano.

    Args:
        config: ``AudioConfig`` con sample_rate, vad_energy_threshold,
                vad_min_speech_ms y vad_padding_ms.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica VAD y devuelve la forma de onda recortada.

        Args:
            waveform: ``[B, T]`` normalizado en [-1, 1].
            lengths : ``[B]`` longitudes reales (muestras).

        Returns:
            waveform_out : ``[B, T_new]`` con padding a la longitud máxima
                           del batch tras VAD.
            new_lengths  : ``[B]`` nuevas longitudes en muestras.
        """
        sr = self.cfg.sample_rate
        batch_size = waveform.size(0)
        device = waveform.device

        if lengths is None:
            lengths = torch.full(
                (batch_size,), waveform.size(1), dtype=torch.long, device=device
            )

        new_waves = []
        new_lengths = []

        for b in range(batch_size):
            wav = waveform[b, : lengths[b]]
            mask = self._compute_speech_mask(wav, sr)
            trimmed = self._apply_mask(wav, mask, sr)
            new_waves.append(trimmed)
            new_lengths.append(trimmed.size(0))

        # Padding al máximo del batch
        max_len = max(new_lengths) if new_lengths else waveform.size(1)
        out = torch.zeros(batch_size, max_len, device=device, dtype=waveform.dtype)
        for b, w in enumerate(new_waves):
            out[b, : w.size(0)] = w

        lengths_tensor = torch.tensor(new_lengths, dtype=torch.long, device=device)
        return out, lengths_tensor

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _compute_speech_mask(
        self,
        wav: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """
        Calcula máscara de voz a nivel de frame (True = voz, False = silencio).

        El frame_length del VAD coincide con el hop_length del análisis
        espectral (10 ms) para que la máscara sea coherente con los frames
        de características.

        Returns:
            mask : ``[T_frames]`` BoolTensor.
        """
        frame_len = int(sr * self.cfg.hop_length_ms / 1000)   # 10 ms
        eps = 1e-10

        # --- Energía de corta duración (STE) ---
        # Padding para divisón en frames exactos
        n_frames = (wav.size(0) + frame_len - 1) // frame_len
        padded_len = n_frames * frame_len
        wav_pad = torch.nn.functional.pad(wav, (0, padded_len - wav.size(0)))
        frames = wav_pad.view(n_frames, frame_len)            # [T_frames, frame_len]

        # STE normalizada (entre 0 y 1)
        ste = frames.pow(2).mean(dim=1)                       # [T_frames]
        ste_norm = ste / (ste.max().clamp(min=eps))

        # --- Tasa de cruce por cero (ZCR) ---
        # Útil para detectar fricativas /s/, /z/, /θ/ típicas del castellano
        signs = (frames[:, 1:] * frames[:, :-1]) < 0         # cruce por cero
        zcr = signs.float().mean(dim=1)                       # [T_frames]
        # Las fricativas sordas tienen ZCR > 0.3 pero STE baja
        # → las incluimos si ZCR es alta (sibilante)
        sibilant = zcr > 0.25

        # Máscara inicial: energía > umbral O sibilante detectada
        mask = (ste_norm > self.cfg.vad_energy_threshold) | sibilant

        # --- Suavizado temporal (fill gaps cortos, quitar silencios cortos) ---
        mask = self._smooth_mask(mask, sr, frame_len)

        return mask

    def _smooth_mask(
        self,
        mask: torch.Tensor,
        sr: int,
        frame_len: int,
    ) -> torch.Tensor:
        """
        Suaviza la máscara VAD:
          - Rellena silencios cortos (< vad_min_speech_ms / 2) dentro de voz.
          - Elimina segmentos de voz demasiado cortos (< vad_min_speech_ms).
        """
        min_frames = max(1, int(sr * self.cfg.vad_min_speech_ms / 1000 / frame_len))

        mask_np = mask.cpu().numpy().astype(bool)
        n = len(mask_np)

        # Rellenar silencios cortos entre segmentos de voz
        i = 0
        while i < n:
            if not mask_np[i]:
                j = i
                while j < n and not mask_np[j]:
                    j += 1
                silence_len = j - i
                if silence_len < min_frames // 2:
                    mask_np[i:j] = True
            i += 1

        # Eliminar segmentos de voz demasiado cortos
        i = 0
        while i < n:
            if mask_np[i]:
                j = i
                while j < n and mask_np[j]:
                    j += 1
                seg_len = j - i
                if seg_len < min_frames:
                    mask_np[i:j] = False
            i += 1

        return torch.tensor(mask_np, dtype=torch.bool, device=mask.device)

    def _apply_mask(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """
        Recorta la forma de onda según la máscara, añadiendo un margen
        (padding_ms) antes y después de cada segmento de voz.
        """
        frame_len = int(sr * self.cfg.hop_length_ms / 1000)
        pad_frames = max(1, int(sr * self.cfg.vad_padding_ms / 1000 / frame_len))
        n = len(mask)

        # Ampliar máscara con margen
        extended = mask.clone()
        for i in range(n):
            if mask[i]:
                start = max(0, i - pad_frames)
                end = min(n, i + pad_frames + 1)
                extended[start:end] = True

        # Índices de muestras
        speech_samples = []
        for i, active in enumerate(extended):
            if active:
                s = i * frame_len
                e = min((i + 1) * frame_len, wav.size(0))
                speech_samples.append(wav[s:e])

        if speech_samples:
            return torch.cat(speech_samples)
        # Si el VAD no detecta nada, devolver la señal completa
        return wav
