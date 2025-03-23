"""
Sinlib: A comprehensive library for Sinhala text processing.

This library provides tools for tokenization, romanization, and transliteration
of Sinhala text, along with various preprocessing utilities.

Available Classes:
    - Tokenizer: For tokenizing Sinhala text
    - Romanizer: For converting Sinhala text to Roman characters
    - Transliterator: For transliterating between scripts
"""

from os import path
from typing import List

from sinlib.romanize import Romanizer
from sinlib.tokenizer import Tokenizer
from sinlib.transliterate import Transliterator
from sinlib.utils import preprocessing
from sinlib.spellcheck import TypoDetector

__all__: List[str] = [
    "Tokenizer",
    "preprocessing",
    "Romanizer",
    "Transliterator",
    "TypoDetector"
]

__version__ = "0.1.9"
