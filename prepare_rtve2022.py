#!/usr/bin/env python3
"""
Descarga y prepara automáticamente un subconjunto de RTVE 2022 (OpenSLR-128).

Hace todo en un solo comando:
  - Descarga el split elegido (por defecto: dev1, el más pequeño)
  - Extrae con validación de rutas segura
  - Busca audios y transcripciones
  - Genera manifest.tsv (audio_path<TAB>transcripción)
  - Genera tokenizer_corpus.txt (una transcripción por línea)

Ejemplo rápido (descarga ~11 horas de dev1):
  python prepare_rtve2022.py --split dev1 --output_dir ./data/rtve2022 --max_samples 1000
Luego entrena:
  python train.py --corpus_dir ./data/rtve2022/manifest.tsv --tokenizer_corpus ./data/rtve2022/tokenizer_corpus --output_dir ./checkpoints --fp16
"""

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


# URLs públicos conocidos de OpenSLR-128 (RTVE 2022)
RTVE_URLS = {
    "dev1": "https://www.openslr.org/resources/128/rtve2022_dev1.tar.gz",
    "dev2": "https://www.openslr.org/resources/128/rtve2022_dev2.tar.gz",
    "train": "https://www.openslr.org/resources/128/rtve2022_train.tar.gz",
    "test": "https://www.openslr.org/resources/128/rtve2022_test.tar.gz",
}


def human_size(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024:
            return f"{num:,.1f}{unit}"
        num /= 1024
    return f"{num:,.1f}TB"


def download(url: str, dest: Path, timeout: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] Archivo ya presente: {dest}")
        return

    print(f"[descargando] {url}")
    try:
        with urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
            total = r.length or 0
            read = 0
            block = 1024 * 1024
            while True:
                chunk = r.read(block)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = read * 100 / total
                    sys.stdout.write(f"\r  {human_size(read)} / {human_size(total)} ({pct:5.1f}%)")
                else:
                    sys.stdout.write(f"\r  {human_size(read)}")
                sys.stdout.flush()
        sys.stdout.write("\n")
    except (HTTPError, URLError, TimeoutError) as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise RuntimeError(f"Extracción insegura detectada en {member.name}")
    tar.extractall(path=path)


def extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if marker.exists():
        print(f"[skip] Ya extraído en {dest}")
        return dest

    print(f"[extrayendo] {archive} -> {dest}")
    with tarfile.open(archive, "r:*") as tar:
        safe_extract(tar, dest)
    marker.touch()
    return dest


def parse_text_files(text_files: List[Path]) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for txt in text_files:
        try:
            with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "\t" in line:
                        utt, text = line.split("\t", 1)
                    else:
                        parts = line.split(maxsplit=1)
                        if len(parts) != 2:
                            continue
                        utt, text = parts
                    transcripts[utt] = text.strip()
        except Exception as exc:
            print(f"[aviso] No se pudo leer {txt}: {exc}")
    return transcripts


def build_pairs(root: Path, max_samples: Optional[int]) -> Tuple[List[Tuple[Path, str]], Path]:
    audio_map: Dict[str, Path] = {}
    for ext in ("*.wav", "*.flac"):
        for wav in root.rglob(ext):
            audio_map[wav.stem] = wav

    text_files = list(root.rglob("text")) + list(root.rglob("*.txt")) + list(root.rglob("*.stm"))
    transcripts = parse_text_files(text_files)

    pairs: List[Tuple[Path, str]] = []
    for utt, wav_path in audio_map.items():
        text = transcripts.get(utt)
        if not text:
            sidecar = wav_path.with_suffix(".txt")
            if sidecar.exists():
                text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            pairs.append((wav_path.resolve(), text))

    pairs.sort(key=lambda x: str(x[0]))
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs, root


def write_outputs(pairs: List[Tuple[Path, str]], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.tsv"
    tokenizer_dir = out_dir / "tokenizer_corpus"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_text = tokenizer_dir / "rtve2022.txt"

    with open(manifest, "w", encoding="utf-8") as mf, open(tokenizer_text, "w", encoding="utf-8") as tf:
        for wav_path, text in pairs:
            mf.write(f"{wav_path}\t{text}\n")
            tf.write(text.strip() + "\n")
    return manifest, tokenizer_dir


def checksum(path: Path, algorithm: str = "sha256") -> str:
    """Checksum de integridad (no para seguridad)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Descarga y prepara RTVE 2022 para entrenamiento.")
    parser.add_argument("--split", choices=RTVE_URLS.keys(), default="dev1",
                        help="Split a descargar (dev1 es el más ligero).")
    parser.add_argument("--output_dir", default="./data/rtve2022",
                        help="Directorio de salida para datos y manifiestos.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Límite opcional de ejemplos para pruebas rápidas.")
    parser.add_argument("--keep_archive", action="store_true",
                        help="No borrar el .tar.gz tras extraer.")
    args = parser.parse_args()

    url = RTVE_URLS[args.split]
    output_dir = Path(args.output_dir).resolve()
    downloads_dir = output_dir / "downloads"
    archive_name = url.split("/")[-1]
    archive_path = downloads_dir / archive_name

    print(f"==> Split: {args.split}")
    print(f"==> Descarga en: {archive_path}")
    download(url, archive_path)
    print(f"[ok] Descargado ({human_size(archive_path.stat().st_size)}) sha256={checksum(archive_path)[:12]}…")

    extracted_root = extract(archive_path, output_dir / "raw" / args.split)

    pairs, data_root = build_pairs(extracted_root, args.max_samples)
    if not pairs:
        raise SystemExit("No se encontraron pares audio+texto después de la extracción.")
    manifest, tokenizer_dir = write_outputs(pairs, output_dir)

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()

    print("\nListo ✅")
    print(f"  Ejemplos: {len(pairs):,}")
    print(f"  Manifest : {manifest}")
    print(f"  Texto BPE: {tokenizer_dir}")
    print("\nEntrenar ejemplo:")
    print(f"  python train.py --corpus_dir {manifest} --tokenizer_corpus {tokenizer_dir} --output_dir ./checkpoints --fp16")


if __name__ == "__main__":
    main()
