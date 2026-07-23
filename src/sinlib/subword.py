from __future__ import annotations

import re
from typing import Any, List, Optional, Union
from sinlib.utils.preprocessing import process_text
from sinlib.encoding import BatchEncoding


def phonological_to_bpe_input(text: str) -> str:
    """
    Prepare text for BPE training/encoding by segmenting into phonological units
    and prepending the special character ' ' to the first unit of each word.
    """
    words = text.split(" ")
    bpe_words = []
    for word in words:
        if not word:
            continue
        units = process_text(word)
        if units:
            units[0] = " " + units[0]
            bpe_words.append(" ".join(units))
    return " ".join(bpe_words)


def bpe_output_to_text(decoded: str) -> str:
    """
    Convert the decoded BPE string back to normal text.
    """
    no_spaces = decoded.replace(" ", "")
    with_spaces = no_spaces.replace(" ", " ")
    return with_spaces.strip()


class SubwordTokenizer:
    """
    Subword tokenizer for Sinhala text.

    Uses the HuggingFace ``tokenizers`` library to build a subword model (BPE)
    on top of base phonological units (consonant + diacritic) rather than raw
    Unicode characters.

    Parameters
    ----------
    hf_tokenizer : Any
        An instance of HuggingFace's ``tokenizers.Tokenizer``.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        self._hf_tokenizer = hf_tokenizer

    @classmethod
    def train_from_corpus(
        cls,
        corpus: List[str],
        vocab_size: int = 5000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ) -> SubwordTokenizer:
        """
        Train a subword BPE tokenizer on a Sinhala corpus.

        Parameters
        ----------
        corpus : list of str
            Training text corpus.
        vocab_size : int, optional
            Target vocabulary size. Default ``5000``.
        min_frequency : int, optional
            Minimum frequency for a subword token to be kept. Default ``2``.
        special_tokens : list of str, optional
            Special tokens list. Default ``["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]``.

        Returns
        -------
        SubwordTokenizer
            A trained subword tokenizer instance.
        """
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError:
            raise ImportError(
                "The 'tokenizers' package is required for SubwordTokenizer. "
                "Please install it via `pip install tokenizers` or `pip install sinlib[subword]`."
            )

        if special_tokens is None:
            special_tokens = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

        # Pre-process the corpus to BPE-compatible format
        prepared_corpus = [phonological_to_bpe_input(text) for text in corpus]

        # Initialize base BPE model
        hf_tok = HFTokenizer(BPE(unk_token="<|unk|>"))
        hf_tok.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

        hf_tok.train_from_iterator(prepared_corpus, trainer=trainer)
        return cls(hf_tok)

    @classmethod
    def from_file(cls, path: str) -> SubwordTokenizer:
        """
        Load a subword tokenizer from a saved vocabulary file.

        Parameters
        ----------
        path : str
            Path to the saved JSON tokenizer file.

        Returns
        -------
        SubwordTokenizer
            Loaded subword tokenizer.
        """
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError:
            raise ImportError(
                "The 'tokenizers' package is required for SubwordTokenizer. "
                "Please install it via `pip install tokenizers` or `pip install sinlib[subword]`."
            )
        hf_tok = HFTokenizer.from_file(path)
        return cls(hf_tok)

    def save(self, path: str) -> None:
        """
        Save the tokenizer vocabulary to a file.

        Parameters
        ----------
        path : str
            Path to save the JSON tokenizer file.
        """
        self._hf_tokenizer.save(path)

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into subword token IDs.

        Parameters
        ----------
        text : str
            Input text to encode.

        Returns
        -------
        list of int
            Subword token IDs.
        """
        prepared = phonological_to_bpe_input(text)
        return self._hf_tokenizer.encode(prepared).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode subword token IDs back to a string.

        Parameters
        ----------
        ids : list of int
            Token IDs to decode.
        skip_special_tokens : bool, optional
            Whether to skip special tokens in the output string. Default ``True``.

        Returns
        -------
        str
            Decoded Sinhala string.
        """
        decoded = self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return bpe_output_to_text(decoded)
