"""
Subsampling CNN 8× para el encoder Conformer.

Inspirado en FastConformer (NVIDIA, 2023) y Conformer (Google, 2020).

El submuestreo 8× reduce T_frames a T_frames/8, lo que disminuye
drásticamente el coste cuadrático de la atención y alinea la resolución
temporal con el paso a paso del decoder.

Arquitectura:
  Conv2D(1→ch, 3×3, stride 2) → GELU → LayerNorm
  Conv2D(ch→ch, 3×3, stride 2) → GELU → LayerNorm
  Conv2D(ch→ch, 3×3, stride 2) → GELU → LayerNorm
  Flatten sobre frecuencia → Linear(ch * F', d_model)

Tres convoluciones con stride=2 → 2³ = 8× reducción en tiempo.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class ConvSubsampling(nn.Module):
    """
    Submuestreo convolucional 8× con proyección lineal final.

    Args:
        in_channels : Normalmente 1 (espectrograma como imagen).
        in_freq     : Dimensión frecuencial de la entrada (n_mels = 80).
        out_dim     : Dimensión de salida (d_model = 512).
        mid_channels: Canales intermedios de las Conv2D.
        dropout     : Dropout tras la proyección lineal.
    """

    STRIDE = 2
    NUM_CONV = 3  # 2^3 = 8×
    KERNEL = 3    # kernel size de cada Conv2d
    PADDING = 1   # padding de cada Conv2d

    def __init__(
        self,
        in_freq: int = 80,
        out_dim: int = 512,
        mid_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_freq = in_freq
        self.out_dim = out_dim

        # Tres bloques Conv2D con stride 2 en la dimensión temporal
        layers = []
        in_ch = 1
        for _ in range(self.NUM_CONV):
            layers += [
                nn.Conv2d(
                    in_ch, mid_channels,
                    kernel_size=self.KERNEL,
                    stride=self.STRIDE,
                    padding=self.PADDING,
                ),
                nn.GELU(),
            ]
            in_ch = mid_channels
        self.conv_stack = nn.Sequential(*layers)

        # Calcular dimensión frecuencial tras las 3 convoluciones
        freq_out = self._compute_freq_out(in_freq)
        linear_in = mid_channels * freq_out

        self.linear = nn.Linear(linear_in, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x       : ``[B, T, n_mels]`` — features log-Mel.
            lengths : ``[B]``             — longitudes en frames.

        Returns:
            out     : ``[B, T', d_model]`` con T' ≈ T / 8.
            lengths : ``[B]``             — longitudes actualizadas.
        """
        # Añadir canal → [B, 1, T, F]
        x = x.unsqueeze(1)

        # Conv2D stack: [B, ch, T', F']
        x = self.conv_stack(x)

        B, ch, T_out, F_out = x.shape

        # Merge canal + frecuencia → [B, T', ch*F']
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T_out, ch * F_out)

        # Proyección lineal → [B, T', d_model]
        x = self.dropout(self.norm(self.linear(x)))

        # Actualizar longitudes
        new_lengths = self._update_lengths(lengths)

        return x, new_lengths

    # ------------------------------------------------------------------

    def _compute_freq_out(self, freq: int) -> int:
        for _ in range(self.NUM_CONV):
            freq = math.floor(
                (freq + 2 * self.PADDING - self.KERNEL) / self.STRIDE + 1
            )
        return freq

    def _update_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths.float()
        for _ in range(self.NUM_CONV):
            out = torch.floor(
                (out + 2 * self.PADDING - self.KERNEL) / self.STRIDE + 1
            )
        return out.long()
