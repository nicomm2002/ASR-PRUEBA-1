"""
ASR España — Sistema de Reconocimiento Automático del Habla
para el español de España (español peninsular).

Arquitectura inspirada en:
  - Whisper (OpenAI, 2022)         https://arxiv.org/abs/2212.04356
  - Conformer (Google, 2020)       https://arxiv.org/abs/2005.08100
  - FastConformer (NVIDIA, 2023)   https://arxiv.org/abs/2305.05084
  - wav2vec 2.0 (Meta, 2020)       https://arxiv.org/abs/2006.11477
  - Omnilingual ASR (Meta, 2025)   https://arxiv.org/abs/2511.09690
  - Samba-ASR (2025)               https://arxiv.org/abs/2501.02832
"""

from .model import ASREspanaModel
from .config import ASREspanaConfig

__all__ = ["ASREspanaModel", "ASREspanaConfig"]
__version__ = "0.1.0"
