"""
Normalizador de texto para español peninsular (España).

Objetivos:
  - Preservar ñ, á, é, í, ó, ú, ü, diéresis
  - Conservar mayúsculas propias (nombres propios, siglas)
  - Mantener dígrafos ll, ch como unidades ortográficas
  - Conservar la distinción s / z / c (ceceo/seseo NO presente en el
    español peninsular normativo; en España septentrional /θ/ ≠ /s/)
  - Eliminar caracteres no deseados manteniendo ortografía española
  - Normalizar números (texto) y abreviaturas comunes del BOE/prensa

Corpus objetivo: El País, El Mundo, BOE, subtítulos RTVE.
"""

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Tabla de sustituciones comunes en prensa/BOE española
# ---------------------------------------------------------------------------

_ABREVIACIONES = {
    r"\bArt\.\s*(\d+)": r"Artículo \1",
    r"\barts?\.\s*(\d+)": r"artículo \1",
    r"\bSr\.\s+": "Señor ",
    r"\bSra\.\s+": "Señora ",
    r"\bDr\.\s+": "Doctor ",
    r"\bDra\.\s+": "Doctora ",
    r"\bD\.\s+": "Don ",
    r"\bD\.ª\s+": "Doña ",
    r"\bEtc\.": "etcétera",
    r"\bpág\.\s*(\d+)": r"página \1",
    r"\bvol\.\s*(\d+)": r"volumen \1",
    r"\bn[.°º]\s*(\d+)": r"número \1",
}

# Dígitos a palabras (solo dígitos sueltos hasta 20 para simplificar)
_DIGITS = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
    "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve",
    "10": "diez", "11": "once", "12": "doce", "13": "trece",
    "14": "catorce", "15": "quince", "16": "dieciséis",
    "17": "diecisiete", "18": "dieciocho", "19": "diecinueve",
    "20": "veinte",
}


class SpanishTextNormalizer:
    """
    Normalizador de texto para el español de España.

    Args:
        lowercase       : Si True convierte todo a minúsculas.
                          Por defecto False (conservar mayúsculas propias).
        expand_numbers  : Si True expande dígitos a palabras (básico).
        expand_abbrev   : Si True expande abreviaturas comunes.
        unicode_compose : Si True aplica NFC (normalización Unicode canónica).
    """

    def __init__(
        self,
        lowercase: bool = False,
        expand_numbers: bool = True,
        expand_abbrev: bool = True,
        unicode_compose: bool = True,
    ):
        self.lowercase = lowercase
        self.expand_numbers = expand_numbers
        self.expand_abbrev = expand_abbrev
        self.unicode_compose = unicode_compose

        # Compilar patrones de abreviaciones
        self._abbrev_patterns = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in _ABREVIACIONES.items()
        ]

        # Caracteres permitidos en español peninsular
        # Letras (incluye ñ, acentuadas, ü), cifras, puntuación básica
        self._allowed = re.compile(
            r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s.,;:!?¡¿\-'\"()\[\]]"
        )

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Normaliza una cadena de texto en español peninsular.

        Args:
            text: Texto crudo (puede contener HTML, caracteres raros, etc.).

        Returns:
            Texto normalizado listo para tokenizar.
        """
        if not text:
            return ""

        # 1. Unicode NFC
        if self.unicode_compose:
            text = unicodedata.normalize("NFC", text)

        # 2. Eliminar etiquetas HTML básicas
        text = re.sub(r"<[^>]+>", " ", text)

        # 3. Expandir abreviaturas
        if self.expand_abbrev:
            for pattern, repl in self._abbrev_patterns:
                text = pattern.sub(repl, text)

        # 4. Expandir números sueltos
        if self.expand_numbers:
            text = self._expand_digits(text)

        # 5. Normalizar espacios alrededor de puntuación
        text = re.sub(r"\s+([.,;:!?¿¡])", r"\1", text)
        text = re.sub(r"([¿¡])\s+", r"\1", text)

        # 6. Eliminar caracteres no permitidos en español
        text = self._allowed.sub(" ", text)

        # 7. Colapsar espacios múltiples
        text = re.sub(r"\s+", " ", text).strip()

        # 8. Minúsculas opcionales
        if self.lowercase:
            text = text.lower()

        return text

    # ------------------------------------------------------------------
    # Expansión de dígitos
    # ------------------------------------------------------------------

    def _expand_digits(self, text: str) -> str:
        """
        Expande números de 0-20 a su forma textual en español.
        Los números mayores se dejan como están (para el BPE).
        """
        def replacer(m: re.Match) -> str:
            num = m.group(0)
            return _DIGITS.get(num, num)

        # Solo números de 1-2 dígitos que no estén pegados a texto
        return re.sub(r"\b\d{1,2}\b", replacer, text)

    # ------------------------------------------------------------------
    # Helpers de limpieza para corpus
    # ------------------------------------------------------------------

    def clean_corpus_line(self, line: str) -> Optional[str]:
        """
        Limpia una línea de corpus para entrenamiento del BPE.
        Devuelve None si la línea es demasiado corta o ruidosa.
        """
        line = self.normalize(line)
        # Filtros de calidad mínima
        if len(line) < 10:
            return None
        # Ratio de caracteres españoles válidos
        valid_chars = re.findall(
            r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]", line
        )
        if len(valid_chars) / max(len(line), 1) < 0.70:
            return None
        return line
