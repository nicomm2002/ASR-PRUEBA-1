"""
Modelo ASR España completo.

Integra las tres fases del sistema:
  Fase 1 — Frontend de audio     (AudioFrontend)
  Fase 2 — Encoder Conformer     (ConformerEncoder)
  Fase 3 — Tokenizador BPE       (SpanishBPETokenizer) [externo al forward]
  +        Decoder híbrido       (TransformerDecoder + CTCDecoder)

Diagrama de flujo:
  audio [B, T_audio]
    │
    ▼ AudioFrontend
  log_mel [B, T_frames, 80]
    │
    ▼ ConformerEncoder
  enc_out [B, T_enc, 512]
    │
    ├──▶ CTCDecoder → ctc_log_probs [T_enc, B, vocab]
    │
    └──▶ TransformerDecoder → att_logits [B, T_dec, vocab]

Loss = (1-λ)·CrossEntropy(att_logits, targets) + λ·CTCLoss
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ASREspanaConfig
from .frontend import AudioFrontend
from .encoder import ConformerEncoder
from .decoder import TransformerDecoder, CTCDecoder


class ASREspanaModel(nn.Module):
    """
    Modelo ASR para el español de España.

    Arquitectura:
      - Frontend:   resampling 16 kHz, Mel 80, CMVN
      - Encoder:    Conformer 17 capas, d=512, subsampling 8×, RoPE
      - Decoder:    Transformer 6 capas + CTC head
      - Tokenizador:BPE peninsular 6 000 tokens (debe cargarse externamente)

    Args:
        config    : ``ASREspanaConfig`` con todos los hiperparámetros.
        vocab_size: Tamaño del vocabulario BPE. Si None se usa
                    config.tokenizer.vocab_size.
    """

    def __init__(self, config: ASREspanaConfig, vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config

        vocab_size = vocab_size or config.tokenizer.vocab_size
        config.vocab_size = vocab_size

        # ---- Fase 1: Frontend de audio ----
        self.frontend = AudioFrontend(config.audio)

        # ---- Fase 2: Encoder Conformer ----
        self.encoder = ConformerEncoder(config.conformer)

        # ---- Decoder híbrido ----
        self.ctc_decoder = CTCDecoder(
            d_model=config.conformer.d_model,
            vocab_size=vocab_size,
            dropout=config.decoder.dropout,
        )
        self.att_decoder = TransformerDecoder(
            config=config.decoder,
            vocab_size=vocab_size,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Inicialización de pesos
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Inicializa los pesos lineales con Xavier uniforme."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward (entrenamiento)
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_sample_rate: int = 16_000,
        audio_lengths: Optional[torch.Tensor] = None,
        apply_vad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo para entrenamiento.

        Args:
            waveform        : ``[B, T_audio]`` audio de entrada.
            tgt_ids         : ``[B, T_dec]`` IDs de tokens objetivo
                              (incluye <sos> al inicio, <eos> al final).
            src_sample_rate : Frecuencia de muestreo del audio de entrada.
            audio_lengths   : ``[B]`` longitudes en muestras (opcional).
            apply_vad       : Si True aplica VAD.

        Returns:
            Diccionario con:
              ``ctc_log_probs``  : ``[T_enc, B, vocab_size]``
              ``att_logits``     : ``[B, T_dec-1, vocab_size]``
              ``enc_lengths``    : ``[B]`` longitudes encoder (post-subsampling)
        """
        # Fase 1: Features
        log_mel, frame_lengths = self.frontend(
            waveform,
            src_sample_rate=src_sample_rate,
            lengths=audio_lengths,
            apply_vad=apply_vad,
        )

        # Fase 2: Encoder
        enc_out, enc_lengths = self.encoder(log_mel, frame_lengths)

        # CTC head
        ctc_log_probs = self.ctc_decoder(enc_out)

        # Decoder de atención (teacher forcing)
        # tgt_ids: [B, T_dec] = [<sos>, tok1, tok2, ..., <eos>]
        # Input:   tgt_ids[:, :-1]   (sin <eos>)
        # Target:  tgt_ids[:, 1:]    (sin <sos>)
        dec_input = tgt_ids[:, :-1]
        att_logits = self.att_decoder(dec_input, enc_out, enc_lengths)

        return {
            "ctc_log_probs": ctc_log_probs,
            "att_logits": att_logits,
            "enc_lengths": enc_lengths,
        }

    # ------------------------------------------------------------------
    # Cálculo del loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        tgt_ids: torch.Tensor,
        tgt_lengths: torch.Tensor,
        pad_id: int = 0,
        blank_id: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula el loss híbrido CTC + Atención.

        Args:
            outputs    : Diccionario devuelto por ``forward``.
            tgt_ids    : ``[B, T_dec]`` IDs objetivo completos (con <sos> y <eos>).
            tgt_lengths: ``[B]`` longitudes reales (sin padding).
            pad_id     : ID del token <pad>.
            blank_id   : ID del token <blank> para CTC.

        Returns:
            Diccionario con ``loss``, ``loss_ctc``, ``loss_att``.
        """
        ctc_lp = outputs["ctc_log_probs"]      # [T_enc, B, V]
        att_logits = outputs["att_logits"]      # [B, T_dec-1, V]
        enc_lengths = outputs["enc_lengths"]    # [B]

        B = tgt_ids.size(0)
        lam = self.config.decoder.ctc_weight

        # ---- CTC Loss ----
        # Targets sin <sos> y <eos>, solo el contenido
        ctc_targets = tgt_ids[:, 1:]  # quitar <sos>
        # Quitar <eos> de cada secuencia
        ctc_target_list = []
        ctc_target_lengths = []
        for b in range(B):
            t = ctc_targets[b]
            # Longitud sin padding y sin eos
            real_len = tgt_lengths[b] - 2  # descontar <sos> y <eos>
            real_len = max(real_len, 1)
            ctc_target_list.append(t[:real_len])
            ctc_target_lengths.append(real_len)

        ctc_targets_flat = torch.cat(ctc_target_list)
        ctc_target_lengths_t = torch.tensor(
            ctc_target_lengths, dtype=torch.long, device=tgt_ids.device
        )

        loss_ctc = F.ctc_loss(
            ctc_lp,
            ctc_targets_flat,
            enc_lengths,
            ctc_target_lengths_t,
            blank=blank_id,
            reduction="mean",
            zero_infinity=True,
        )

        # ---- Attention Loss ----
        # att_logits: [B, T_dec-1, V]
        # Target:     tgt_ids[:, 1:]  (sin <sos>)
        att_target = tgt_ids[:, 1:]
        B_, T_, V = att_logits.shape
        loss_att = F.cross_entropy(
            att_logits.reshape(B_ * T_, V),
            att_target.reshape(B_ * T_),
            ignore_index=pad_id,
            label_smoothing=self.config.decoder.label_smoothing,
        )

        # ---- Loss híbrido ----
        loss = (1 - lam) * loss_att + lam * loss_ctc

        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att,
        }

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transcribe(
        self,
        waveform: torch.Tensor,
        src_sample_rate: int = 16_000,
        sos_id: int = 1,
        eos_id: int = 2,
        max_len: int = 200,
        apply_vad: bool = True,
    ) -> torch.Tensor:
        """
        Transcribe audio a secuencia de IDs de tokens.

        Args:
            waveform        : ``[B, T_audio]`` o ``[T_audio]``.
            src_sample_rate : Frecuencia de muestreo.
            sos_id, eos_id  : IDs de inicio/fin de secuencia.
            max_len         : Longitud máxima de la transcripción.
            apply_vad       : Aplicar VAD antes de la transcripción.

        Returns:
            ``[B, T_pred]`` IDs predichos.
        """
        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Frontend + Encoder
        log_mel, frame_lengths = self.frontend(
            waveform,
            src_sample_rate=src_sample_rate,
            apply_vad=apply_vad,
        )
        enc_out, enc_lengths = self.encoder(log_mel, frame_lengths)

        # Greedy decode
        return self.att_decoder.greedy_decode(
            enc_out, enc_lengths, sos_id=sos_id, eos_id=eos_id, max_len=max_len
        )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Cuenta los parámetros del modelo."""
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)
