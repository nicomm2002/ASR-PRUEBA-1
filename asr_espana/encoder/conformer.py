"""
Encoder Conformer para ASR en español de España.

Arquitectura basada en:
  Gulati et al., "Conformer: Convolution-augmented Transformer for
  Speech Recognition" (Google, 2020). https://arxiv.org/abs/2005.08100

Con mejoras de:
  Rekesh et al., "Fast Conformer with Linearly Scalable Attention for
  Efficient Speech Recognition" (NVIDIA, 2023). https://arxiv.org/abs/2305.05084

Cada bloque Conformer aplica el siguiente patrón (Macaron-Net):
  x = x + 0.5 · FFN(x)
  x = x + MHSA(x)           ← con RoPE
  x = x + ConvModule(x)
  x = x + 0.5 · FFN(x)
  x = LayerNorm(x)

Pipeline completo del encoder:
  [B, T, 80]  →  ConvSubsampling 8×  →  [B, T/8, d_model]
                 17 × ConformerBlock
                 LayerNorm final
              →  [B, T/8, d_model]
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ConformerConfig
from .rope import RotaryPositionalEncoding
from .subsampling import ConvSubsampling


# ---------------------------------------------------------------------------
# Feed-Forward Module
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """
    Módulo feed-forward del bloque Conformer.

    Linear(d, 4d) → SiLU → Dropout → Linear(4d, d) → Dropout
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention con RoPE
# ---------------------------------------------------------------------------

class MultiHeadSelfAttentionRoPE(nn.Module):
    """
    Atención multi-cabeza con Rotary Positional Encoding.

    Soporta máscara de padding para batches con longitudes variables.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible entre num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        self.rope = RotaryPositionalEncoding(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x                : ``[B, T, d_model]``
            key_padding_mask : ``[B, T]`` BoolTensor — True donde hay padding.

        Returns:
            ``[B, T, d_model]``
        """
        B, T, _ = x.shape

        residual = x
        x = self.norm(x)

        # Q, K, V: [B, T, d_model] → [B, num_heads, T, head_dim]
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        # Aplicar RoPE
        q, k = self.rope.apply_rotary(q, k)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, T]

        if key_padding_mask is not None:
            # Expandir máscara: [B, 1, 1, T]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Context: [B, H, T, head_dim] → [B, T, d_model]
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        out = self.out_drop(self.out_proj(context))
        return residual + out


# ---------------------------------------------------------------------------
# Convolution Module del Conformer
# ---------------------------------------------------------------------------

class ConvolutionModule(nn.Module):
    """
    Módulo de convolución del bloque Conformer.

    LayerNorm → PointwiseConv(d, 2d) → GLU → DepthwiseConv(kernel) →
    BatchNorm → SiLU → PointwiseConv(d, d) → Dropout
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size debe ser impar"
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x                : ``[B, T, d_model]``
            key_padding_mask : ``[B, T]`` para enmascarar padding.
        """
        residual = x
        x = self.norm(x)

        # Transponer para Conv1D: [B, d_model, T]
        x = x.transpose(1, 2)

        # Pointwise Conv + GLU
        x = self.pw_conv1(x)        # [B, 2*d, T]
        x = F.glu(x, dim=1)         # [B, d, T]

        # Depthwise Conv + BN + SiLU
        x = self.dw_conv(x)
        x = self.bn(x)
        x = F.silu(x)

        # Pointwise Conv final
        x = self.pw_conv2(x)        # [B, d, T]
        x = self.dropout(x)

        # Transponer de vuelta: [B, T, d_model]
        x = x.transpose(1, 2)

        # Enmascarar posiciones de padding para no contaminar la normalización
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return residual + x


# ---------------------------------------------------------------------------
# Bloque Conformer completo
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    """
    Bloque Conformer completo con patrón Macaron-Net.

      x = x + 0.5 · FFN₁(x)
      x = x + MHSA(x)
      x = x + ConvModule(x)
      x = x + 0.5 · FFN₂(x)
      x = LayerNorm(x)
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        d = config.d_model

        self.ff1 = FeedForwardModule(d, config.ff_expansion_factor, config.dropout)
        self.attn = MultiHeadSelfAttentionRoPE(
            d, config.num_heads, config.attention_dropout, config.rope_base
        )
        self.conv = ConvolutionModule(d, config.conv_kernel_size, config.dropout)
        self.ff2 = FeedForwardModule(d, config.ff_expansion_factor, config.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = self.attn(x, key_padding_mask=key_padding_mask)
        x = self.conv(x, key_padding_mask=key_padding_mask)
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Encoder completo
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """
    Encoder Conformer completo.

    Pipeline:
      [B, T, 80]
        → ConvSubsampling 8×   → [B, T/8, d_model]
        → 17 × ConformerBlock
        → LayerNorm
        → [B, T/8, d_model]

    Args:
        config: ``ConformerConfig`` con todos los hiperparámetros.
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        # Subsampling CNN 8×
        self.subsampling = ConvSubsampling(
            in_freq=config.input_dim,
            out_dim=config.d_model,
            mid_channels=config.subsampling_channels,
            dropout=config.dropout,
        )

        # 17 bloques Conformer
        self.blocks = nn.ModuleList(
            [ConformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(config.d_model)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x       : ``[B, T, n_mels]`` features log-Mel.
            lengths : ``[B]`` longitudes en frames antes del subsampling.

        Returns:
            enc_out : ``[B, T', d_model]``
            lengths : ``[B]`` longitudes tras el subsampling 8×.
        """
        # Subsampling 8×
        x, lengths = self.subsampling(x, lengths)  # [B, T', d_model]

        # Máscara de padding para la atención
        key_padding_mask = self._make_padding_mask(x, lengths)

        # Bloques Conformer
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        return x, lengths

    # ------------------------------------------------------------------

    @staticmethod
    def _make_padding_mask(
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Crea máscara booleana [B, T] — True en posiciones de padding."""
        B, T, _ = x.shape
        if lengths.min().item() == T:
            return None
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask  # [B, T]
