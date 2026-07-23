"""
Sinlib: A comprehensive library for Sinhala text processing.

Provides tools for tokenization, spell-checking, romanization, and
transliteration of Sinhala text, along with preprocessing utilities.

Available Classes
-----------------
Tokenizer
    Character-level tokenizer for Sinhala text. Mirrors the HuggingFace
    ``PreTrainedTokenizer`` interface — use ``Tokenizer.from_pretrained()``
    to load the default pretrained vocabulary.

BatchEncoding
    Dict-like container returned by the tokenizer. Supports attribute access
    (``enc.input_ids``, ``enc.attention_mask``) and dict-style access.

TypoDetector
    N-gram–based spell checker for Sinhala. Use
    ``TypoDetector.from_pretrained()`` or instantiate directly.

preprocessing
    Utility module exposing ``remove_english_characters``,
    ``remove_non_printable``, and ``get_sinhala_character_ratio``.

Romanizer
    Converts Sinhala text to Roman (Latin) script.

Transliterator
    ML-based Sinhala → Roman transliteration using a pre-trained BiLSTM.
    *Currently disabled from default __all__ pending model vocabulary realignment.*
"""

from typing import List

from sinlib.encoding import BatchEncoding
from sinlib.tokenizer import Tokenizer
from sinlib.subword import SubwordTokenizer
from sinlib.spellcheck import TypoDetector
from sinlib.romanize import Romanizer
from sinlib.transliterate import Transliterator  # noqa: F401
from sinlib.utils import preprocessing

__all__: List[str] = [
    "BatchEncoding",
    "Tokenizer",
    "SubwordTokenizer",
    "TypoDetector",
    "Romanizer",
    "preprocessing",
]

__version__ = "0.1.13"
