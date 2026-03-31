"""
Decoder para ASR en español de España.

Arquitectura híbrida:
  1. CTC Head   — para entrenamiento rápido y beam search eficiente.
  2. Transformer Decoder — para rescoring y generación autoregresiva.

Loss híbrida:
  L = (1 - λ) · L_att  +  λ · L_ctc        con λ = 0.3 por defecto

Referencia:
  Hori et al., "Joint CTC/Attention-based End-to-End Speech Recognition
  Using Multi-task Learning" (2017). https://arxiv.org/abs/1609.06773
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DecoderConfig
from ..encoder.rope import RotaryPositionalEncoding


# ---------------------------------------------------------------------------
# CTC Decoder (cabeza lineal)
# ---------------------------------------------------------------------------

class CTCDecoder(nn.Module):
    """
    Cabeza CTC: proyección lineal sobre las salidas del encoder.

    Args:
        d_model    : Dimensión del encoder.
        vocab_size : Tamaño del vocabulario (incluyendo <blank>).
        dropout    : Dropout antes de la proyección.
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_out: ``[B, T, d_model]``

        Returns:
            log_probs: ``[T, B, vocab_size]`` — formato requerido por
                       ``torch.nn.CTCLoss``.
        """
        x = self.drop(encoder_out)
        logits = self.linear(x)           # [B, T, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)  # [T, B, vocab_size]


# ---------------------------------------------------------------------------
# Transformer Decoder (con cross-attention)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Capa del Transformer Decoder con:
      - Masked self-attention (autoregresiva) + RoPE
      - Cross-attention sobre el encoder
      - Feed-forward
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        d = config.d_model
        h = config.num_heads
        head_dim = d // h
        ff_hidden = d * config.ff_expansion_factor

        # Masked self-attention
        self.self_norm = nn.LayerNorm(d)
        self.self_qkv = nn.Linear(d, 3 * d, bias=False)
        self.self_out = nn.Linear(d, d, bias=False)
        self.self_drop = nn.Dropout(config.dropout)
        self.self_attn_drop = nn.Dropout(config.dropout)
        self.rope = RotaryPositionalEncoding(head_dim)

        # Cross-attention
        self.cross_norm = nn.LayerNorm(d)
        self.cross_q = nn.Linear(d, d, bias=False)
        self.cross_kv = nn.Linear(d, 2 * d, bias=False)
        self.cross_out = nn.Linear(d, d, bias=False)
        self.cross_drop = nn.Dropout(config.dropout)
        self.cross_attn_drop = nn.Dropout(config.dropout)

        # Feed-forward
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ff_hidden, d),
            nn.Dropout(config.dropout),
        )

        self.num_heads = h
        self.head_dim = head_dim
        self.d_model = d
        self.scale = math.sqrt(head_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x                    : ``[B, T_dec, d_model]`` (decoder input)
            encoder_out          : ``[B, T_enc, d_model]``
            tgt_mask             : ``[T_dec, T_dec]`` causal mask (float, -inf)
            src_key_padding_mask : ``[B, T_enc]`` padding mask del encoder
        """
        B, T_dec, _ = x.shape
        H, HD = self.num_heads, self.head_dim

        # ---- Masked self-attention ----
        residual = x
        x = self.self_norm(x)
        qkv = self.self_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T_dec, H, HD).transpose(1, 2) for t in qkv]
        q, k = self.rope.apply_rotary(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if tgt_mask is not None:
            scores = scores + tgt_mask.unsqueeze(0).unsqueeze(0)
        weights = self.self_attn_drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T_dec, -1)
        x = residual + self.self_drop(self.self_out(ctx))

        # ---- Cross-attention ----
        residual = x
        x_norm = self.cross_norm(x)
        T_enc = encoder_out.size(1)
        q_c = self.cross_q(x_norm).view(B, T_dec, H, HD).transpose(1, 2)
        kv = self.cross_kv(encoder_out).chunk(2, dim=-1)
        k_c, v_c = [t.view(B, T_enc, H, HD).transpose(1, 2) for t in kv]
        scores_c = torch.matmul(q_c, k_c.transpose(-2, -1)) / self.scale
        if src_key_padding_mask is not None:
            scores_c = scores_c.masked_fill(
                src_key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        weights_c = self.cross_attn_drop(F.softmax(scores_c, dim=-1))
        ctx_c = torch.matmul(weights_c, v_c).transpose(1, 2).contiguous().view(B, T_dec, -1)
        x = residual + self.cross_drop(self.cross_out(ctx_c))

        # ---- Feed-forward ----
        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Decoder Transformer con N capas.

    Args:
        config    : ``DecoderConfig``.
        vocab_size: Tamaño del vocabulario (incluye especiales).
    """

    def __init__(self, config: DecoderConfig, vocab_size: int):
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.embed_scale = math.sqrt(config.d_model)
        self.embed_drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        # Compartir pesos embedding / proyección de salida (weight tying)
        self.output_proj.weight = self.embed.weight

    # ------------------------------------------------------------------

    def forward(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        enc_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt_ids    : ``[B, T_dec]`` IDs de tokens objetivo.
            encoder_out: ``[B, T_enc, d_model]``
            enc_lengths: ``[B]`` longitudes del encoder (para padding mask).

        Returns:
            logits: ``[B, T_dec, vocab_size]``
        """
        B, T_dec = tgt_ids.shape
        device = tgt_ids.device

        # Máscara causal
        tgt_mask = self._causal_mask(T_dec, device)

        # Padding mask del encoder
        src_pad_mask = None
        if enc_lengths is not None:
            T_enc = encoder_out.size(1)
            src_pad_mask = (
                torch.arange(T_enc, device=device).unsqueeze(0) >= enc_lengths.unsqueeze(1)
            )

        # Embedding + escala
        x = self.embed(tgt_ids) * self.embed_scale
        x = self.embed_drop(x)

        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask)

        x = self.norm(x)
        return self.output_proj(x)

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Máscara causal triangular inferior (valores -inf en la parte superior)."""
        mask = torch.full((sz, sz), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    # ------------------------------------------------------------------
    # Greedy decode (inferencia)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 200,
    ) -> torch.Tensor:
        """
        Decodificación greedy autoregresiva.

        Args:
            encoder_out: ``[B, T, d_model]``
            enc_lengths: ``[B]``
            sos_id     : ID del token <sos>.
            eos_id     : ID del token <eos>.
            max_len    : Longitud máxima de salida.

        Returns:
            preds: ``[B, T_dec]`` IDs predichos (sin <sos>).
        """
        B = encoder_out.size(0)
        device = encoder_out.device

        # Inicializar con <sos>
        dec_input = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        predictions = []
        for _ in range(max_len):
            logits = self.forward(dec_input, encoder_out, enc_lengths)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            predictions.append(next_token)
            dec_input = torch.cat([dec_input, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return torch.cat(predictions, dim=1)  # [B, T_dec]
