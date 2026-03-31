"""
Script de entrenamiento para ASR España.

Uso:
  python train.py --corpus_dir /datos/corpus_es \
                  --tokenizer_corpus /datos/texto_peninsular \
                  --output_dir ./checkpoints \
                  [--batch_size 16] [--lr 5e-4] [--max_steps 500000]

Corpus de audio esperado:
  El directorio --corpus_dir debe contener pares:
    *.wav (audio 16 kHz mono)  +  *.txt (transcripción)
  o un archivo manifest.tsv con columnas:  audio_path  transcription

Corpus de texto para BPE:
  --tokenizer_corpus: directorio con archivos .txt (uno por línea)
  Fuentes recomendadas:
    - El País / El Mundo (noticias)
    - BOE (Boletín Oficial del Estado)
    - Subtítulos RTVE
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from asr_espana import ASREspanaModel, ASREspanaConfig
from asr_espana.config import AudioConfig, ConformerConfig, TokenizerConfig, DecoderConfig
from asr_espana.tokenizer import SpanishBPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpanishASRDataset(Dataset):
    """
    Dataset para ASR en español.

    Formatos soportados:
      1. Directorio con pares audio.wav + audio.txt
      2. Archivo manifest.tsv (audio_path \\t transcription)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: SpanishBPETokenizer,
        max_audio_len_s: float = 30.0,
        sample_rate: int = 16_000,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_audio_len = int(max_audio_len_s * sample_rate)
        self.samples: List[Tuple[str, str]] = []
        self._load(data_path)

    def _load(self, data_path: str) -> None:
        p = Path(data_path)
        if p.is_file() and p.suffix in (".tsv", ".csv"):
            sep = "\t" if p.suffix == ".tsv" else ","
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        self.samples.append((parts[0], parts[1]))
        elif p.is_dir():
            for wav_path in sorted(p.glob("**/*.wav")):
                txt_path = wav_path.with_suffix(".txt")
                if txt_path.exists():
                    self.samples.append((str(wav_path), txt_path.read_text(encoding="utf-8").strip()))
        logger.info(f"Dataset cargado: {len(self.samples):,} muestras desde {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        audio_path, text = self.samples[idx]
        try:
            waveform = self._load_audio(audio_path)
        except Exception as e:
            logger.warning(f"Error cargando {audio_path}: {e}")
            return None

        token_ids = self.tokenizer.encode(text, add_sos=True, add_eos=True)
        return {
            "waveform": waveform,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "audio_len": waveform.size(0),
            "text": text,
        }

    def _load_audio(self, path: str) -> torch.Tensor:
        try:
            import torchaudio
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=False)
            else:
                wav = wav.squeeze(0)
            if sr != self.sample_rate:
                import torchaudio.functional as TAF
                wav = TAF.resample(wav, sr, self.sample_rate)
        except ImportError:
            # Fallback: leer WAV sin torchaudio (solo PCM 16-bit)
            import wave, struct
            with wave.open(path, "rb") as wf:
                n = wf.getnframes()
                raw = wf.readframes(n)
            samples = struct.unpack(f"{n}h", raw)
            wav = torch.tensor(samples, dtype=torch.float32) / 32768.0

        # Truncar si es demasiado largo
        if wav.size(0) > self.max_audio_len:
            wav = wav[: self.max_audio_len]
        return wav


def collate_fn(batch):
    """Padding dinámico de audio y tokens."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_audio = max(b["audio_len"] for b in batch)
    max_tokens = max(b["token_ids"].size(0) for b in batch)

    B = len(batch)
    waveforms = torch.zeros(B, max_audio)
    token_ids = torch.zeros(B, max_tokens, dtype=torch.long)
    audio_lens = torch.zeros(B, dtype=torch.long)
    token_lens = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        a_len = item["audio_len"]
        t_len = item["token_ids"].size(0)
        waveforms[i, :a_len] = item["waveform"]
        token_ids[i, :t_len] = item["token_ids"]
        audio_lens[i] = a_len
        token_lens[i] = t_len

    return {
        "waveforms": waveforms,
        "token_ids": token_ids,
        "audio_lens": audio_lens,
        "token_lens": token_lens,
    }


# ---------------------------------------------------------------------------
# Scheduler con warmup lineal
# ---------------------------------------------------------------------------

def get_warmup_scheduler(optimizer, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Config ---
    config = ASREspanaConfig(
        audio=AudioConfig(),
        conformer=ConformerConfig(),
        tokenizer=TokenizerConfig(
            vocab_size=args.vocab_size,
            corpus_path=args.tokenizer_corpus,
        ),
        decoder=DecoderConfig(
            ctc_weight=args.ctc_weight,
        ),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        accumulate_grad_batches=args.grad_accum,
    )

    # --- Tokenizador ---
    tokenizer = SpanishBPETokenizer(config.tokenizer)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer", "bpe_espana")

    if args.train_tokenizer or not Path(tokenizer_path + ".model").exists():
        logger.info("Entrenando tokenizador BPE peninsular…")
        corpus_files = sorted(Path(args.tokenizer_corpus).glob("**/*.txt"))
        if not corpus_files:
            logger.error(f"No se encontraron archivos .txt en {args.tokenizer_corpus}")
            sys.exit(1)
        tokenizer.train(
            corpus_files=[str(f) for f in corpus_files],
            model_prefix=tokenizer_path,
            vocab_size=args.vocab_size,
        )
    else:
        logger.info("Cargando tokenizador existente…")
        tokenizer.load(tokenizer_path)

    vocab_size = tokenizer.vocab_size

    # --- Modelo ---
    model = ASREspanaModel(config, vocab_size=vocab_size).to(device)
    logger.info(
        f"Modelo creado: {model.num_parameters() / 1e6:.1f}M parámetros"
    )

    # --- Dataset y DataLoader ---
    train_ds = SpanishASRDataset(
        args.corpus_dir, tokenizer, sample_rate=config.audio.sample_rate
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # --- Optimizador ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=1e-2,
    )
    scheduler = get_warmup_scheduler(optimizer, args.warmup_steps)
    scaler = GradScaler(enabled=(device.type == "cuda" and args.fp16))

    # --- Cargar checkpoint si existe ---
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumiendo desde step {start_step}")

    # --- Bucle de entrenamiento ---
    model.train()
    global_step = start_step
    optimizer.zero_grad()
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        for batch in train_dl:
            if batch is None:
                continue
            if global_step >= args.max_steps:
                break

            waveforms = batch["waveforms"].to(device)
            token_ids = batch["token_ids"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            token_lens = batch["token_lens"].to(device)

            with autocast(enabled=(device.type == "cuda" and args.fp16)):
                outputs = model(
                    waveforms,
                    tgt_ids=token_ids,
                    audio_lengths=audio_lens,
                )
                losses = model.compute_loss(
                    outputs,
                    tgt_ids=token_ids,
                    tgt_lengths=token_lens,
                    pad_id=tokenizer.pad_id,
                    blank_id=tokenizer.blank_id,
                )
                loss = losses["loss"] / args.grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Step {global_step:6d}/{args.max_steps} | "
                    f"Loss {losses['loss'].item():.4f} | "
                    f"CTC {losses['loss_ctc'].item():.4f} | "
                    f"Att {losses['loss_att'].item():.4f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.0f}s"
                )
                t0 = time.time()

            # Guardado de checkpoint
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f"checkpoint_step{global_step:06d}.pt"
                )
                torch.save({
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                }, ckpt_path)
                logger.info(f"Checkpoint guardado: {ckpt_path}")

        if global_step >= args.max_steps:
            break

    # Guardar modelo final
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save({"step": global_step, "model": model.state_dict()}, final_path)
    logger.info(f"Entrenamiento finalizado. Modelo guardado en {final_path}")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo ASR España"
    )
    # Datos
    parser.add_argument("--corpus_dir", required=True,
                        help="Directorio con pares audio+transcripción o manifest.tsv")
    parser.add_argument("--tokenizer_corpus", required=True,
                        help="Directorio con archivos .txt de texto peninsular (para BPE)")
    parser.add_argument("--output_dir", default="./checkpoints",
                        help="Directorio de salida para checkpoints y tokenizador")
    parser.add_argument("--train_tokenizer", action="store_true",
                        help="Forzar reentrenamiento del tokenizador BPE")

    # Modelo
    parser.add_argument("--vocab_size", type=int, default=6000,
                        help="Tamaño vocabulario BPE")
    parser.add_argument("--ctc_weight", type=float, default=0.3,
                        help="Peso del loss CTC en el loss híbrido (0-1)")

    # Entrenamiento
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10_000)
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Acumulación de gradientes")
    parser.add_argument("--fp16", action="store_true",
                        help="Entrenamiento con precisión mixta (AMP)")
    parser.add_argument("--num_workers", type=int, default=4)

    # Logging / checkpointing
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a un checkpoint para reanudar el entrenamiento")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
