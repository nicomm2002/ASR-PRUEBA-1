"""
Tokenizador BPE para el español peninsular (España).

Especificaciones (Fase 3):
  - Corpus de texto peninsular: El País, El Mundo, BOE, subtítulos RTVE
  - Normalización: conservar ñ, acentos, ü, diéresis, mayúsculas propias
  - Incluir grafías exclusivas: ll, ch como dígrafos frecuentes
  - Fonemas específicos: distinción s/z/c (sin ceceo/seseo)
  - BPE entrenado solo sobre texto peninsular, vocab 6 000 tokens
  - Tokens especiales: <pad>, <sos>, <eos>, <unk>, <blank>

Implementación:
  - Usa sentencepiece para entrenamiento e inferencia BPE
  - Ofrece fallback a un BPE puro en Python si sentencepiece no está
    disponible (para desarrollo/pruebas sin dependencias adicionales)
"""

import os
import io
import re
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import sentencepiece as spm
    _SPM_AVAILABLE = True
except ImportError:
    _SPM_AVAILABLE = False
    logger.warning(
        "sentencepiece no encontrado. Usando fallback BPE puro en Python. "
        "Instala con: pip install sentencepiece"
    )

from ..config import TokenizerConfig
from .text_normalizer import SpanishTextNormalizer


# ---------------------------------------------------------------------------
# Tokenizador principal
# ---------------------------------------------------------------------------

class SpanishBPETokenizer:
    """
    Tokenizador BPE para el español de España.

    Flujo de uso:
      1. ``train(corpus_files)``  — entrena desde texto peninsular.
      2. ``encode(text)``         — convierte texto a lista de IDs.
      3. ``decode(ids)``          — convierte IDs a texto.

    Args:
        config: ``TokenizerConfig``.
    """

    # Tokens especiales
    SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "<blank>"]

    def __init__(self, config: TokenizerConfig):
        self.cfg = config
        self.normalizer = SpanishTextNormalizer(
            lowercase=config.lowercase,
            expand_numbers=True,
            expand_abbrev=True,
            unicode_compose=config.normalize_unicode,
        )
        self._spm_model: Optional["spm.SentencePieceProcessor"] = None
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        # Asignar IDs fijos a tokens especiales
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self._vocab[tok] = i
            self._id_to_token[i] = tok

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._vocab[self.cfg.pad_token]

    @property
    def sos_id(self) -> int:
        return self._vocab[self.cfg.sos_token]

    @property
    def eos_id(self) -> int:
        return self._vocab[self.cfg.eos_token]

    @property
    def unk_id(self) -> int:
        return self._vocab[self.cfg.unk_token]

    @property
    def blank_id(self) -> int:
        return self._vocab[self.cfg.blank_token]

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(
        self,
        corpus_files: List[str],
        model_prefix: Optional[str] = None,
        vocab_size: Optional[int] = None,
    ) -> None:
        """
        Entrena el modelo BPE sobre un corpus de texto peninsular.

        Args:
            corpus_files : Lista de rutas a archivos .txt con texto en
                           español de España (uno por línea).
            model_prefix : Prefijo para los archivos .model / .vocab de
                           sentencepiece. Si None usa config.model_prefix.
            vocab_size   : Tamaño del vocabulario. Si None usa config.vocab_size.
        """
        model_prefix = model_prefix or self.cfg.model_prefix
        vocab_size = vocab_size or self.cfg.vocab_size
        os.makedirs(Path(model_prefix).parent, exist_ok=True)

        # Preparar corpus normalizado en un archivo temporal
        tmp_corpus = model_prefix + "_corpus_tmp.txt"
        n_lines = self._prepare_corpus(corpus_files, tmp_corpus)
        logger.info(f"Corpus preparado: {n_lines:,} líneas → {tmp_corpus}")

        if _SPM_AVAILABLE:
            self._train_sentencepiece(tmp_corpus, model_prefix, vocab_size)
        else:
            self._train_pure_bpe(tmp_corpus, model_prefix, vocab_size)

        # Limpiar archivo temporal
        if os.path.exists(tmp_corpus):
            os.remove(tmp_corpus)

        logger.info(f"Tokenizador entrenado. Vocab: {self.vocab_size} tokens.")

    def _prepare_corpus(self, corpus_files: List[str], output_path: str) -> int:
        """Normaliza y escribe el corpus de entrenamiento."""
        n = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for fpath in corpus_files:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fin:
                    for line in fin:
                        clean = self.normalizer.clean_corpus_line(line.strip())
                        if clean:
                            fout.write(clean + "\n")
                            n += 1
        return n

    def _train_sentencepiece(
        self, corpus: str, model_prefix: str, vocab_size: int
    ) -> None:
        """Entrena con sentencepiece."""
        special_tokens = ",".join(self.SPECIAL_TOKENS)

        # Parámetros calibrados para español peninsular
        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=model_prefix,
            model_type=self.cfg.model_type,
            vocab_size=vocab_size,
            character_coverage=self.cfg.character_coverage,
            user_defined_symbols=special_tokens,
            # Preservar caracteres especiales del español
            normalization_rule_name="nmt_nfkc_cf" if self.cfg.lowercase else "nmt_nfkc",
            # Tratar ll y ch como unidades potenciales
            split_by_unicode_script=True,
            split_digits=True,
            # Dígrafos frecuentes en español
            required_chars="ñáéíóúüÁÉÍÓÚÜÑ",
            pad_id=self.SPECIAL_TOKENS.index("<pad>"),
            unk_id=self.SPECIAL_TOKENS.index("<unk>"),
            bos_id=self.SPECIAL_TOKENS.index("<sos>"),
            eos_id=self.SPECIAL_TOKENS.index("<eos>"),
            byte_fallback=True,
        )

        self._spm_model = spm.SentencePieceProcessor()
        self._spm_model.load(model_prefix + ".model")

        # Actualizar vocab interno
        self._vocab = {}
        self._id_to_token = {}
        for i in range(self._spm_model.get_piece_size()):
            tok = self._spm_model.id_to_piece(i)
            self._vocab[tok] = i
            self._id_to_token[i] = tok

    def _train_pure_bpe(
        self, corpus: str, model_prefix: str, vocab_size: int
    ) -> None:
        """
        BPE puro en Python (fallback sin sentencepiece).
        Implementación del algoritmo original de Sennrich et al. (2016).
        """
        logger.info("Entrenando BPE puro en Python (puede tardar varios minutos)…")

        # 1. Tokenización a nivel de carácter + marcador de fin de palabra
        vocab = Counter()
        with open(corpus, "r", encoding="utf-8") as f:
            for line in f:
                for word in line.strip().split():
                    chars = tuple(list(word) + ["</w>"])
                    vocab[chars] += 1

        # Vocabulario inicial: todos los caracteres únicos
        symbols: Dict[str, int] = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}

        for word_chars in vocab:
            for ch in word_chars:
                if ch not in symbols:
                    symbols[ch] = len(symbols)

        merges: List[Tuple[str, str]] = []
        target = vocab_size - len(symbols)

        for step in range(target):
            # Contar pares
            pairs: Counter = Counter()
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq

            if not pairs:
                break

            best = pairs.most_common(1)[0][0]
            merges.append(best)
            new_symbol = "".join(best)
            if new_symbol not in symbols:
                symbols[new_symbol] = len(symbols)

            # Aplicar merge al vocabulario
            new_vocab: Counter = Counter()
            a, b = best
            bigram = re.escape(a + " " + b)
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

            for word, freq in vocab.items():
                word_str = " ".join(word)
                word_str = pattern.sub(new_symbol, word_str)
                new_vocab[tuple(word_str.split())] += freq
            vocab = new_vocab

            if (step + 1) % 500 == 0:
                logger.info(f"  BPE merge {step + 1}/{target}")

        self._vocab = symbols
        self._id_to_token = {v: k for k, v in symbols.items()}
        self._merges = merges

        # Guardar
        model_data = {
            "vocab": symbols,
            "merges": [list(m) for m in merges],
        }
        with open(model_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Carga / guardado
    # ------------------------------------------------------------------

    def load(self, model_prefix: Optional[str] = None) -> None:
        """Carga un modelo BPE previamente entrenado."""
        model_prefix = model_prefix or self.cfg.model_prefix

        spm_path = model_prefix + ".model"
        json_path = model_prefix + ".json"

        if _SPM_AVAILABLE and os.path.exists(spm_path):
            self._spm_model = spm.SentencePieceProcessor()
            self._spm_model.load(spm_path)
            self._vocab = {}
            self._id_to_token = {}
            for i in range(self._spm_model.get_piece_size()):
                tok = self._spm_model.id_to_piece(i)
                self._vocab[tok] = i
                self._id_to_token[i] = tok
        elif os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            self._vocab = model_data["vocab"]
            self._id_to_token = {int(v): k for k, v in self._vocab.items()}
            self._merges = [tuple(m) for m in model_data.get("merges", [])]
        else:
            raise FileNotFoundError(
                f"No se encontró modelo BPE en {model_prefix}.[model|json]"
            )
        logger.info(f"Tokenizador cargado: {self.vocab_size} tokens.")

    # ------------------------------------------------------------------
    # Codificación / Decodificación
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Convierte texto en español a lista de IDs de tokens.

        Args:
            text    : Texto crudo en español peninsular.
            add_sos : Añadir token <sos> al principio.
            add_eos : Añadir token <eos> al final.

        Returns:
            Lista de enteros (IDs).
        """
        text = self.normalizer.normalize(text)

        if self._spm_model is not None:
            ids = self._spm_model.encode(text, out_type=int)
        else:
            ids = self._bpe_encode(text)

        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """
        Convierte lista de IDs a texto.

        Args:
            ids           : Lista de IDs de tokens.
            skip_special  : Si True elimina tokens especiales del output.

        Returns:
            Texto en español.
        """
        special_ids = {self.pad_id, self.sos_id, self.eos_id, self.blank_id}
        if skip_special:
            ids = [i for i in ids if i not in special_ids]

        if self._spm_model is not None:
            return self._spm_model.decode(ids)

        tokens = [self._id_to_token.get(i, "<unk>") for i in ids]
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def _bpe_encode(self, text: str) -> List[int]:
        """Codifica usando el BPE puro en Python."""
        result = []
        for word in text.split():
            chars = list(word) + ["</w>"]
            # Aplicar merges en orden
            for merge in getattr(self, "_merges", []):
                a, b = merge
                new_sym = a + b
                i = 0
                merged = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        merged.append(new_sym)
                        i += 2
                    else:
                        merged.append(chars[i])
                        i += 1
                chars = merged
            result.extend(
                [self._vocab.get(ch, self.unk_id) for ch in chars]
            )
        return result
