# ASR España — Reconocimiento Automático del Habla en Español Peninsular

Sistema ASR (*Automatic Speech Recognition*) diseñado desde cero para el **español de España** (español peninsular), implementado en Python con PyTorch.

---

## Arquitectura

### Fase 1 — Frontend de audio (`asr_espana/frontend/`)

| Paso | Descripción |
|------|-------------|
| Resampling | 16 kHz mono |
| Normalización | Amplitud normalizada por utterance |
| VAD | Ajustado a patrones prosódicos del español peninsular (ritmo silábico, fricativas /s/θ/) |
| Pre-énfasis | `y[t] = x[t] − 0.97·x[t−1]` |
| Framing | Ventanas 25 ms, salto 10 ms |
| Ventana | Hann |
| STFT | N=512 |
| Banco Mel | 80 filtros, 80–7600 Hz |
| Log-compresión | `log(max(mel, 1e-10))` |
| CMVN | Por utterance |
| **Salida** | **`[batch, T, 80]`** |

### Fase 2 — Encoder Conformer (`asr_espana/encoder/`)

| Componente | Detalle |
|------------|---------|
| Proyección | Linear 80 → 512 (d_model) |
| Subsampling | CNN 3× (stride 2) = **8×** reducción temporal |
| Bloques | **17 Conformer** (Macaron-Net: ½FFN → MHSA → Conv → ½FFN → LN) |
| Atención | Multi-Head Self-Attention (8 cabezas) + **RoPE** |
| Convolución | Depthwise Conv1D, kernel 31 |
| LayerNorm | Final |
| **Salida** | **`[batch, T/8, 512]`** |

### Fase 3 — Tokenizador BPE Peninsular (`asr_espana/tokenizer/`)

- **Corpus**: El País, El Mundo, BOE, subtítulos RTVE
- **Preserva**: ñ, á, é, í, ó, ú, ü, mayúsculas propias
- **Dígrafos**: `ll`, `ch` como unidades frecuentes
- **Distinción fonémica**: /s/ ≠ /θ/ (sin ceceo ni seseo)
- **BPE** entrenado solo sobre texto peninsular
- **Vocabulario**: 6 000 tokens
- **Tokens especiales**: `<pad>`, `<sos>`, `<eos>`, `<unk>`, `<blank>`

### Decoder híbrido (`asr_espana/decoder/`)

- **CTC head**: proyección lineal sobre el encoder para CTC loss
- **Transformer Decoder**: 6 capas con cross-attention + RoPE
- **Loss híbrido**: `L = 0.7·L_att + 0.3·L_ctc`
- **Weight tying**: embedding compartido con proyección de salida

---

## Referencias

| Paper | Institución | Año |
|-------|-------------|-----|
| [Whisper](https://arxiv.org/abs/2212.04356) — Robust Speech Recognition via Large-Scale Weak Supervision | OpenAI | 2022 |
| [Conformer](https://arxiv.org/abs/2005.08100) — Convolution-augmented Transformer for Speech Recognition | Google | 2020 |
| [FastConformer](https://arxiv.org/abs/2305.05084) — Fast Conformer with Linearly Scalable Attention | NVIDIA | 2023 |
| [wav2vec 2.0](https://arxiv.org/abs/2006.11477) — Self-Supervised Learning of Speech Representations | Meta | 2020 |
| [Omnilingual ASR](https://arxiv.org/abs/2511.09690) — Open-Source Multilingual Speech Recognition for 1600+ Languages | Meta | 2025 |
| [Samba-ASR](https://arxiv.org/abs/2501.02832) — State-of-the-Art Speech Recognition with Mamba SSMs | — | 2025 |

---

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias:
- `torch >= 2.0`
- `torchaudio >= 2.0`
- `sentencepiece >= 0.1.99`

---

## Uso

### 1. Entrenar el tokenizador BPE y el modelo

```bash
python train.py \
  --corpus_dir /datos/audio_es \
  --tokenizer_corpus /datos/texto_peninsular \
  --output_dir ./checkpoints \
  --vocab_size 6000 \
  --batch_size 16 \
  --fp16
```

El directorio `--corpus_dir` debe contener pares `*.wav` + `*.txt`  
o un archivo `manifest.tsv` con columnas: `audio_path<TAB>transcripción`.

El directorio `--tokenizer_corpus` debe contener archivos `.txt` con  
texto peninsular (El País, El Mundo, BOE, RTVE), uno por línea.

### 2. Transcribir audio

```bash
python inference.py \
  --model ./checkpoints/model_final.pt \
  --tokenizer ./checkpoints/tokenizer/bpe_espana \
  --audio audio1.wav audio2.wav
```

### 3. Preparar RTVE 2022 en un solo comando

Descarga y prepara un subconjunto de RTVE 2022 (OpenSLR-128) listo para `train.py`.  
Por defecto usa `dev1`, el split más pequeño.

```bash
python prepare_rtve2022.py \
  --split dev1 \
  --output_dir ./data/rtve2022 \
  --max_samples 1000   # opcional: limitar ejemplos para prueba rápida

# Luego entrena con el manifiesto generado:
python train.py \
  --corpus_dir ./data/rtve2022/manifest.tsv \
  --tokenizer_corpus ./data/rtve2022/tokenizer_corpus \
  --output_dir ./checkpoints \
  --fp16
```

### 4. Uso como biblioteca Python

```python
import torch
from asr_espana import ASREspanaModel, ASREspanaConfig
from asr_espana.tokenizer import SpanishBPETokenizer

# Crear modelo (~136M parámetros)
config = ASREspanaConfig()
model = ASREspanaModel(config, vocab_size=6000)

# Transcribir audio
wav = torch.randn(1, 32000)  # 2 s @ 16 kHz
pred_ids = model.transcribe(wav)
```

---

## Estructura del proyecto

```
asr_espana/
├── __init__.py
├── config.py                   # Todos los hiperparámetros
├── model.py                    # Modelo completo integrado
├── frontend/
│   ├── audio_frontend.py       # Fase 1: pipeline de audio
│   └── vad.py                  # VAD peninsular (STE + ZCR)
├── encoder/
│   ├── conformer.py            # Fase 2: bloques Conformer
│   ├── rope.py                 # Rotary Positional Encoding
│   └── subsampling.py          # CNN subsampling 8×
├── tokenizer/
│   ├── tokenizer.py            # Fase 3: BPE peninsular
│   └── text_normalizer.py      # Normalización texto español ES
└── decoder/
    └── decoder.py              # CTC + Transformer Decoder
train.py                        # Script de entrenamiento
inference.py                    # Script de transcripción
requirements.txt
```

---

## Características específicas del español de España

- **VAD** calibrado para el **ritmo silábico** del castellano (~5.7 síl/s)
- Detección de **fricativas** /s/, /θ/ mediante ZCR
- BPE con **cobertura de caracteres 1.0** para cubrir toda la ortografía española
- `required_chars`: `ñáéíóúüÁÉÍÓÚÜÑ` garantizados en el vocabulario
- Normalización de **abreviaturas BOE**: `Art.`, `n.º`, `Sr.`, etc.
- Sin transformación a minúsculas por defecto → preserva nombres propios y siglas
