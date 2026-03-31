"""
Script de inferencia / transcripción para el modelo ASR España.

Uso:
  python inference.py --model ./checkpoints/model_final.pt \
                      --tokenizer ./checkpoints/tokenizer/bpe_espana \
                      --audio archivo.wav [archivo2.wav ...]
                      [--device cuda] [--max_len 200]

Salida:
  Para cada archivo de audio imprime la transcripción en español peninsular.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import torch

from asr_espana import ASREspanaModel, ASREspanaConfig
from asr_espana.tokenizer import SpanishBPETokenizer
from asr_espana.config import TokenizerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga de audio
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16_000) -> torch.Tensor:
    """
    Carga un archivo de audio y lo convierte a tensor mono 16 kHz.

    Formatos soportados: WAV, FLAC, MP3, OGG (requiere torchaudio).
    """
    try:
        import torchaudio
        import torchaudio.functional as TAF
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=False)
        else:
            wav = wav.squeeze(0)
        if sr != target_sr:
            wav = TAF.resample(wav, sr, target_sr)
        return wav
    except ImportError:
        logger.warning("torchaudio no disponible; usando fallback WAV PCM")
        import wave, struct
        with wave.open(path, "rb") as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
        samples = struct.unpack(f"{n}h", raw)
        return torch.tensor(samples, dtype=torch.float32) / 32768.0


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def transcribe_files(
    audio_paths: List[str],
    model: ASREspanaModel,
    tokenizer: SpanishBPETokenizer,
    device: torch.device,
    max_len: int = 200,
    apply_vad: bool = True,
    batch_size: int = 8,
) -> List[str]:
    """
    Transcribe una lista de archivos de audio.

    Args:
        audio_paths: Lista de rutas a archivos de audio.
        model      : Modelo ASR cargado.
        tokenizer  : Tokenizador BPE.
        device     : Dispositivo de inferencia.
        max_len    : Longitud máxima de tokens de salida.
        apply_vad  : Aplicar VAD antes de la transcripción.
        batch_size : Número de audios procesados simultáneamente.

    Returns:
        Lista de transcripciones (una por audio).
    """
    results = []
    sr = model.config.audio.sample_rate

    for i in range(0, len(audio_paths), batch_size):
        chunk = audio_paths[i: i + batch_size]

        # Cargar audios y hacer padding al máximo del batch
        waveforms = []
        lengths = []
        for path in chunk:
            try:
                wav = load_audio(path, target_sr=sr).to(device)
                waveforms.append(wav)
                lengths.append(wav.size(0))
            except Exception as e:
                logger.error(f"No se pudo cargar {path}: {e}")
                waveforms.append(torch.zeros(sr, device=device))  # 1s de silencio
                lengths.append(sr)

        max_len_audio = max(lengths)
        wav_batch = torch.zeros(len(waveforms), max_len_audio, device=device)
        for j, wav in enumerate(waveforms):
            wav_batch[j, : wav.size(0)] = wav
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)

        # Transcribir
        pred_ids = model.transcribe(
            wav_batch,
            src_sample_rate=sr,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_len,
            apply_vad=apply_vad,
        )

        # Decodificar
        for b in range(pred_ids.size(0)):
            ids = pred_ids[b].tolist()
            text = tokenizer.decode(ids, skip_special=True)
            results.append(text)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcripción de audio con ASR España"
    )
    parser.add_argument("--model", required=True,
                        help="Ruta al checkpoint del modelo (.pt)")
    parser.add_argument("--tokenizer", required=True,
                        help="Prefijo del modelo BPE (sin extensión .model/.json)")
    parser.add_argument("--audio", nargs="+", required=True,
                        help="Archivos de audio a transcribir")
    parser.add_argument("--device", default="auto",
                        help="Dispositivo: auto, cpu, cuda, cuda:0, …")
    parser.add_argument("--max_len", type=int, default=200,
                        help="Longitud máxima de tokens en la transcripción")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size para la inferencia")
    parser.add_argument("--no_vad", action="store_true",
                        help="Desactivar VAD durante la transcripción")
    parser.add_argument("--output", type=str, default=None,
                        help="Archivo de salida (por defecto stdout)")
    args = parser.parse_args()

    # Dispositivo
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Dispositivo: {device}")

    # Tokenizador
    tokenizer_cfg = TokenizerConfig()
    tokenizer = SpanishBPETokenizer(tokenizer_cfg)
    tokenizer.load(args.tokenizer)
    logger.info(f"Tokenizador cargado: {tokenizer.vocab_size} tokens")

    # Modelo
    ckpt = torch.load(args.model, map_location=device)
    config = ckpt.get("config", ASREspanaConfig())
    model = ASREspanaModel(config, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        f"Modelo cargado: {model.num_parameters() / 1e6:.1f}M parámetros"
    )

    # Archivos de audio
    audio_paths = []
    for pattern in args.audio:
        paths = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        audio_paths.extend([str(p) for p in paths if p.exists()])

    if not audio_paths:
        logger.error("No se encontraron archivos de audio")
        sys.exit(1)

    logger.info(f"Transcribiendo {len(audio_paths)} archivo(s)…")

    # Transcribir
    transcriptions = transcribe_files(
        audio_paths,
        model,
        tokenizer,
        device,
        max_len=args.max_len,
        apply_vad=not args.no_vad,
        batch_size=args.batch_size,
    )

    # Salida
    lines = []
    for path, text in zip(audio_paths, transcriptions):
        line = f"{path}\t{text}"
        lines.append(line)
        print(line)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Transcripciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
