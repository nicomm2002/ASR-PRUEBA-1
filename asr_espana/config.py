"""
Configuración central del sistema ASR España.
Todos los hiperparámetros y rutas configurables se definen aquí.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Audio frontend
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int = 16_000          # Hz objetivo tras resampling
    n_fft: int = 512                   # Puntos FFT
    win_length_ms: float = 25.0        # Ventana análisis (ms)
    hop_length_ms: float = 10.0        # Salto de ventana (ms)
    n_mels: int = 80                   # Filtros Mel
    f_min: float = 80.0                # Frecuencia mínima del banco Mel
    f_max: float = 7_600.0             # Frecuencia máxima del banco Mel
    pre_emphasis_coeff: float = 0.97   # Coeficiente pre-énfasis
    cmvn: bool = True                  # CMVN por utterance
    log_floor: float = 1e-10           # Suelo para log-compresión

    # VAD
    vad_energy_threshold: float = 0.05  # Umbral energía normalizada
    vad_min_speech_ms: float = 150.0    # Duración mínima segmento voz (ms)
    vad_padding_ms: float = 80.0        # Margen antes/después del segmento

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.win_length_ms / 1000)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_length_ms / 1000)


# ---------------------------------------------------------------------------
# Encoder Conformer
# ---------------------------------------------------------------------------

@dataclass
class ConformerConfig:
    d_model: int = 512                # Dimensión del modelo
    num_heads: int = 8                # Cabezas de atención
    num_layers: int = 17              # Bloques Conformer
    ff_expansion_factor: int = 4      # Factor expansión feed-forward
    conv_kernel_size: int = 31        # Núcleo convolución Conformer
    subsampling_factor: int = 8       # Factor subsampling CNN (8×)
    subsampling_channels: int = 256   # Canales intermedios subsampling
    dropout: float = 0.1              # Dropout general
    attention_dropout: float = 0.1    # Dropout en atención
    rope_base: float = 10_000.0       # Base RoPE
    input_dim: int = 80               # Dimensión entrada (n_mels)


# ---------------------------------------------------------------------------
# Tokenizador BPE peninsular
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    vocab_size: int = 6_000
    model_type: str = "bpe"           # sentencepiece model type
    character_coverage: float = 1.0   # Cobertura caracteres (español)
    # Tokens especiales
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    blank_token: str = "<blank>"       # CTC blank
    # Normalización
    normalize_unicode: bool = True
    lowercase: bool = False            # Conservar mayúsculas
    keep_accents: bool = True          # ñ, á, é, í, ó, ú, ü
    # Rutas
    model_prefix: str = "tokenizer/bpe_espana"
    corpus_path: Optional[str] = None  # Corpus texto peninsular


# ---------------------------------------------------------------------------
# Decoder (atención + CTC)
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_expansion_factor: int = 4
    dropout: float = 0.1
    max_len: int = 512                 # Longitud máxima secuencia decodificada
    ctc_weight: float = 0.3            # Peso CTC en loss híbrido (0 = solo atención)
    label_smoothing: float = 0.1


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

@dataclass
class ASREspanaConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Entrenamiento
    batch_size: int = 16
    learning_rate: float = 5e-4
    warmup_steps: int = 10_000
    max_steps: int = 500_000
    grad_clip: float = 1.0
    accumulate_grad_batches: int = 4

    # Idioma objetivo
    language: str = "es-ES"           # Español de España

    def __post_init__(self):
        # Sincronizar d_model entre encoder y decoder
        self.decoder.d_model = self.conformer.d_model
        # Sincronizar vocab_size
        self._vocab_size: Optional[int] = None

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, v: int):
        self._vocab_size = v
