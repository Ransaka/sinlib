"""
Optional Sinhala-CharBERT neural backend for sinlib's TypoDetector.

Provides inference-only model code and a high-level ``CharBERTBackend`` that
loads published CharBERT seq2seq checkpoints (HF Hub or local directory).
``torch`` is imported lazily; the base sinlib install does not require it.
"""

from sinlib.charbert.backend import CharBERTBackend
from sinlib.charbert.config import SinhalaCharBERTConfig

__all__ = [
    "CharBERTBackend",
    "SinhalaCharBERTConfig",
]
