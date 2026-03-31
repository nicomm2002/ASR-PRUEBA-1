"""
Rotary Positional Encoding (RoPE).

Implementación basada en:
  Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  https://arxiv.org/abs/2104.09864

RoPE codifica información posicional directamente en las matrices de consulta
(Q) y clave (K) mediante rotaciones, sin añadir vectores externos.  Esto
permite generalización a longitudes mayores que las vistas en entrenamiento y
es la opción de codificación posicional del encoder Conformer de este sistema.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) para secuencias de longitud variable.

    Aplica rotaciones complejas 2D a pares consecutivos de dimensiones de
    ``Q`` y ``K``, de modo que el producto escalar entre dos posiciones
    depende solo de su diferencia relativa.

    Args:
        d_model : Dimensión del modelo (debe ser par).
        base    : Base de la progresión geométrica de frecuencias (10 000).
        max_len : Longitud máxima de caché de cosenos/senos.
    """

    def __init__(
        self,
        d_model: int,
        base: float = 10_000.0,
        max_len: int = 4096,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model debe ser par para RoPE"
        self.d_model = d_model
        self.base = base
        self._build_cache(max_len)

    # ------------------------------------------------------------------

    def _build_cache(self, max_len: int) -> None:
        """Pre-calcula cos y sin para posiciones 0..max_len-1."""
        half = self.d_model // 2
        # θ_i = 1 / base^(2i / d_model)
        theta = 1.0 / (
            self.base ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        positions = torch.arange(max_len, dtype=torch.float32)
        # freqs: [max_len, half]
        freqs = torch.outer(positions, theta)
        # Repetir para las d_model dimensiones
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_len, d_model]
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _get_sin_cos(self, seq_len: int) -> tuple:
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len * 2)
        cos = self.cos_cache[:seq_len]  # [T, d_model]
        sin = self.sin_cache[:seq_len]
        return cos, sin

    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rota la segunda mitad del vector y lo mezcla con la primera."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple:
        """
        Aplica RoPE a los tensores Q y K.

        Args:
            q: ``[B, num_heads, T, head_dim]``
            k: ``[B, num_heads, T, head_dim]``

        Returns:
            q_rot, k_rot con las mismas formas.
        """
        seq_len = q.size(-2)
        cos, sin = self._get_sin_cos(seq_len)
        # Expandir para broadcasting: [1, 1, T, d_model]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna ``x`` sin modificar — RoPE se aplica dentro de la atención.
        Este forward solo existe para compatibilidad de interfaz.
        """
        return x
