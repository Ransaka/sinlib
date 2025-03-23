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

# Define package resources directory
RESOURCES_DIR: str = path.join(path.dirname(__file__), 'data')

__all__: List[str] = [
    "Tokenizer",
    "preprocessing",
    "Romanizer",
    "Transliterator"
]

__version__ = "0.1.6"
