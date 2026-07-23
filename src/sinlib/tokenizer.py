"""
Sinhala text tokenization module.

Provides a character-level tokenizer specifically designed for Sinhala text that
combines base consonants with their vowel diacritics into single meaningful tokens.
The public API mirrors the HuggingFace ``PreTrainedTokenizer`` interface.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, cast
import concurrent.futures

from tqdm.auto import tqdm

from .encoding import BatchEncoding
from .utils.preprocessing import process_text, load_default_vocab_map, load_default_config

# Sentinel used to detect when the caller did not supply max_length
_UNSET = object()

# HuggingFace Hub repo that hosts the default pretrained tokenizer
_DEFAULT_HF_REPO = "Ransaka/sinlib"


class Tokenizer:
    """
    Character-level tokenizer for Sinhala text.

    Combines base consonants with following vowel diacritics into single
    tokens (e.g. ``ග`` + ``ෙ`` → ``ගෙ``), producing linguistically meaningful
    units that are better suited for NLP/ML tasks than naive Unicode splitting.

    The interface mirrors HuggingFace's ``PreTrainedTokenizer`` — you can load
    a pretrained vocabulary with :meth:`from_pretrained`, encode text with
    :meth:`__call__` or :meth:`encode`, and decode back with :meth:`decode`.

    Parameters
    ----------
    model_max_length : int, optional
        Maximum sequence length used for padding/truncation.  Pass ``None``
        to disable length-based operations (useful for the :class:`~sinlib.Romanizer`
        which manages its own sequences).
    unk_token : str, optional
        Special token for unknown characters.  Default ``"<|unk|>"``.
    pad_token : str, optional
        Special token for padding.  Default ``"<|pad|>"``.
    eos_token : str, optional
        End-of-sequence token.  Default ``"<|end_of_text|>"``.
    bos_token : str, optional
        Beginning-of-sequence token.  Default ``"<|bos|>"``.

    Examples
    --------
    Load the default pretrained tokenizer from HuggingFace Hub:

    >>> from sinlib import Tokenizer
    >>> tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
    >>> enc = tokenizer("ආයුබෝවන්")
    >>> enc.input_ids
    [...]

    Train a tokenizer from your own corpus:

    >>> tokenizer = Tokenizer(model_max_length=128)
    >>> tokenizer.train(["මම ගෙදර ගියා", "ඔහු පාසලට ගියා"])
    >>> enc = tokenizer("ගෙදර")
    >>> enc.input_ids
    [...]
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_max_length: Optional[int] = None,
        unk_token: str = "<|unk|>",
        pad_token: str = "<|pad|>",
        eos_token: str = "<|end_of_text|>",
        bos_token: str = "<|bos|>",
        # Legacy alias kept for callers using max_length=
        **kwargs,
    ) -> None:
        # Support legacy kwarg `max_length` transparently
        if "max_length" in kwargs and model_max_length is None:
            model_max_length = kwargs.pop("max_length")

        # Special token strings
        self.unk_token: str = unk_token
        self.pad_token: str = pad_token
        self.eos_token: str = eos_token
        self.bos_token: str = bos_token

        # Legacy aliases so existing code using the old attribute names still works
        self.unknown_token: str = unk_token
        self.end_of_text_token: str = eos_token

        self.special_tokens: List[str] = [
            self.pad_token,
            self.unk_token,
            self.eos_token,
            self.bos_token,
        ]

        # Configuration
        self.model_max_length: Optional[int] = model_max_length
        # Legacy alias
        self.max_length: Optional[int] = model_max_length

        # Token mappings (populated after train() or from_pretrained())
        self.vocab_map: Optional[Dict[str, int]] = None
        self.token_id_to_token_map: Optional[Dict[int, str]] = None

        # Special token IDs (populated after vocab is built)
        self.unk_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None

        # Legacy aliases
        self.unknown_token_id: Optional[int] = None
        self.end_of_text_token_id: Optional[int] = None

        # Internal training state
        self.tokenized_chars: List[str] = []
        self.unique_chars: Set[str] = set()

    # ------------------------------------------------------------------
    # Class-method constructor (HF-style)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        model_max_length: Optional[int] = None,
        **kwargs,
    ) -> "Tokenizer":
        """
        Load a tokenizer from a pretrained model or local directory.

        Accepts either a HuggingFace Hub repository ID (e.g.
        ``"Ransaka/sinlib"``) or a local filesystem path containing
        ``vocab.json`` and ``config.json``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            HuggingFace Hub repo ID **or** local directory path.
        model_max_length : int, optional
            Override the ``model_max_length`` stored in ``config.json``.
        **kwargs
            Additional keyword arguments forwarded to :class:`Tokenizer`.

        Returns
        -------
        Tokenizer
            A fully initialised tokenizer ready for encoding/decoding.

        Raises
        ------
        ValueError
            If the path does not exist or required files are missing.

        Examples
        --------
        Load from HuggingFace Hub:

        >>> tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

        Load from a local directory:

        >>> tokenizer = Tokenizer.from_pretrained("./my_tokenizer/")
        """
        # Determine whether we have a local path or a HF repo ID
        local_path = Path(pretrained_model_name_or_path)
        is_local = local_path.exists() and local_path.is_dir()

        # Build an instance with a temporary max_length; will be overwritten
        # once we load the config.
        instance = cls(model_max_length=model_max_length, **kwargs)

        if is_local:
            instance._load_from_local(local_path)
        else:
            # Treat as HF Hub repo ID.  The default repo "Ransaka/sinlib" is
            # handled by the existing helpers; for other repos we could extend
            # this in the future.
            if pretrained_model_name_or_path == _DEFAULT_HF_REPO or pretrained_model_name_or_path == "sinlib":
                instance._load_default_tokenizer()
            else:
                raise ValueError(
                    f"Repository '{pretrained_model_name_or_path}' is not a local path "
                    f"and is not the default '{_DEFAULT_HF_REPO}' repo.  "
                    "Please pass a valid local directory path or 'Ransaka/sinlib'."
                )

        # Allow caller to override max_length from config
        if model_max_length is not None:
            instance.model_max_length = model_max_length
            instance.max_length = model_max_length

        return instance

    # ------------------------------------------------------------------
    # Encoding — primary public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        add_bos_token: bool = False,
        # Legacy parameter kept for backward-compat
        truncate_and_pad: bool = False,
        allowed_special_tokens: Optional[List[str]] = None,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """
        Encode text into a :class:`~sinlib.BatchEncoding`.

        Accepts a single string or a list of strings.  When a list is
        provided the sequences are padded to the same length when
        ``padding=True``.

        Parameters
        ----------
        text : str or list of str
            The input text(s) to tokenize.
        padding : bool, optional
            Pad all sequences to ``max_length`` (or ``model_max_length``).
            Default ``False``.
        truncation : bool, optional
            Truncate sequences longer than ``max_length`` (or
            ``model_max_length``).  Default ``False``.
        max_length : int, optional
            Override the tokenizer's ``model_max_length`` for this call only.
        add_special_tokens : bool, optional
            Currently unused placeholder for future special-token injection.
            Default ``True``.
        return_attention_mask : bool, optional
            Include an ``"attention_mask"`` field in the output.
            Default ``True``.
        add_bos_token : bool, optional
            Prepend the beginning-of-sequence token.  Default ``False``.
        truncate_and_pad : bool, optional
            Legacy alias for ``padding=True, truncation=True``.
        allowed_special_tokens : list of str, optional
            Special tokens that are allowed to appear in the *input* text and
            should be encoded as their token IDs rather than skipped.
        return_tensors : str, optional
            Currently unused.  Accepted values: ``None``.

        Returns
        -------
        BatchEncoding
            An object with ``.input_ids`` and ``.attention_mask`` fields.
            When the input is a list, each field is a list-of-lists.

        Raises
        ------
        ValueError
            If the tokenizer has not been trained or loaded.

        Examples
        --------
        >>> enc = tokenizer("ගෙදර")
        >>> enc.input_ids
        [10, 11, 12]
        >>> enc.attention_mask
        [1, 1, 1]

        >>> enc = tokenizer(["ගෙදර", "ඔහු"], padding=True)
        >>> enc.input_ids
        [[10, 11, 12], [7, 8, 0]]
        """
        # Legacy flag support
        if truncate_and_pad:
            padding = True
            truncation = True

        if allowed_special_tokens is None:
            allowed_special_tokens = []

        effective_max = max_length or self.model_max_length

        if isinstance(text, list):
            return self._batch_encode_impl(
                text,
                padding=padding,
                truncation=truncation,
                max_length=effective_max,
                return_attention_mask=return_attention_mask,
                add_bos_token=add_bos_token,
                allowed_special_tokens=allowed_special_tokens,
                return_tensors=return_tensors,
            )

        return self._encode_impl(
            text,
            padding=padding,
            truncation=truncation,
            max_length=effective_max,
            return_attention_mask=return_attention_mask,
            add_bos_token=add_bos_token,
            allowed_special_tokens=allowed_special_tokens,
            return_tensors=return_tensors,
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        add_bos_token: bool = False,
        allowed_special_tokens: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Encode a single string into a list of token IDs (no padding/truncation).

        Parameters
        ----------
        text : str
            Input text to encode.
        add_special_tokens : bool, optional
            Placeholder; currently unused.  Default ``True``.
        add_bos_token : bool, optional
            Prepend the BOS token ID.  Default ``False``.
        allowed_special_tokens : list of str, optional
            Special tokens that appear in the text and should be encoded as
            token IDs rather than skipped.

        Returns
        -------
        list of int
            Token IDs without padding or truncation.

        Examples
        --------
        >>> tokenizer.encode("ගෙදර")
        [10, 11, 12]
        """
        if allowed_special_tokens is None:
            allowed_special_tokens = []
        return self._encode_impl(
            text,
            padding=False,
            truncation=False,
            max_length=None,
            return_attention_mask=False,
            add_bos_token=add_bos_token,
            allowed_special_tokens=allowed_special_tokens,
        ).input_ids

    def encode_plus(
        self,
        text: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        add_bos_token: bool = False,
        allowed_special_tokens: Optional[List[str]] = None,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """
        Encode a single string and return a :class:`~sinlib.BatchEncoding`.

        Equivalent to calling the tokenizer directly but always operates on a
        single string.

        Parameters
        ----------
        text : str
            Input text to encode.
        padding : bool, optional
            Pad to ``max_length`` / ``model_max_length``.  Default ``False``.
        truncation : bool, optional
            Truncate at ``max_length`` / ``model_max_length``.  Default ``False``.
        max_length : int, optional
            Overrides ``model_max_length`` for this call.
        return_attention_mask : bool, optional
            Include ``"attention_mask"`` in output.  Default ``True``.
        add_bos_token : bool, optional
            Prepend the BOS token.  Default ``False``.
        allowed_special_tokens : list of str, optional
            Special tokens to encode rather than skip.
        return_tensors : str, optional
            Convert lists to tensors. Supported: ``"pt"``, ``"tf"``, ``"np"``.

        Returns
        -------
        BatchEncoding
            Encoding result with ``.input_ids`` and ``.attention_mask``.

        Examples
        --------
        >>> enc = tokenizer.encode_plus("ගෙදර", padding=True, max_length=10)
        >>> enc.input_ids
        [10, 11, 12, 0, 0, 0, 0, 0, 0, 0]
        """
        if allowed_special_tokens is None:
            allowed_special_tokens = []
        effective_max = max_length or self.model_max_length
        return self._encode_impl(
            text,
            padding=padding,
            truncation=truncation,
            max_length=effective_max,
            return_attention_mask=return_attention_mask,
            add_bos_token=add_bos_token,
            allowed_special_tokens=allowed_special_tokens,
            return_tensors=return_tensors,
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into a list of string tokens without converting to IDs.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list of str
            List of Sinhala character tokens (consonant + diacritic units).

        Examples
        --------
        >>> tokenizer.tokenize("ආයුබෝවන්")
        ['ආ', 'යු', 'බෝ', 'ව', 'න්']
        """
        return process_text(text)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        ids: Union[List[int], "BatchEncoding", Dict[str, List[int]]],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode a sequence of token IDs back to a string.

        Parameters
        ----------
        ids : list of int, BatchEncoding, or dict
            Token IDs to decode.  Accepts:

            * a plain ``list[int]``
            * a :class:`~sinlib.BatchEncoding` (uses ``.input_ids``)
            * a ``dict`` with an ``"input_ids"`` key (legacy)

        skip_special_tokens : bool, optional
            If ``True``, special tokens (pad, unk, eos, bos) are omitted from
            the output.  Default ``False``.

        Returns
        -------
        str
            The decoded text.

        Raises
        ------
        ValueError
            If the tokenizer has not been trained or loaded.

        Examples
        --------
        >>> tokenizer.decode([10, 11, 12])
        'ගෙදර'
        >>> tokenizer.decode([10, 11, 12], skip_special_tokens=True)
        'ගෙදර'
        """
        if not self.token_id_to_token_map:
            raise ValueError("Tokenizer not trained. Call train() or from_pretrained() first.")

        # Unwrap BatchEncoding or dict
        if isinstance(ids, BatchEncoding):
            ids = ids.input_ids
        elif isinstance(ids, dict) and "input_ids" in ids:
            ids = ids["input_ids"]

        special_token_ids: List[int] = [
            self.vocab_map[tok]  # type: ignore[index]
            for tok in self.special_tokens
            if self.vocab_map and tok in self.vocab_map
        ]

        int_ids: List[int] = cast(List[int], ids)
        filtered = [
            token_id for token_id in int_ids
            if not skip_special_tokens or token_id not in special_token_ids
        ]

        id_map: Dict[int, str] = self.token_id_to_token_map  # type: ignore[assignment]
        return "".join(id_map.get(int(tid), self.unk_token) for tid in filtered)

    def batch_decode(
        self,
        batch_ids: Union[List[List[int]], List["BatchEncoding"]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences back to strings.

        Parameters
        ----------
        batch_ids : list of (list of int) or list of BatchEncoding
            Sequences to decode.
        skip_special_tokens : bool, optional
            If ``True``, special tokens are omitted from each output string.
            Default ``False``.

        Returns
        -------
        list of str
            Decoded texts.

        Raises
        ------
        ValueError
            If the tokenizer has not been trained or loaded.
        """
        if not self.token_id_to_token_map:
            raise ValueError("Tokenizer not trained. Call train() or from_pretrained() first.")
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    # ------------------------------------------------------------------
    # Vocabulary helpers (HF-standard)
    # ------------------------------------------------------------------

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """
        Convert token string(s) to their corresponding integer ID(s).

        Parameters
        ----------
        tokens : str or list of str
            A single token string or a list of token strings.

        Returns
        -------
        int or list of int
            The corresponding token ID(s).  Unknown tokens return the
            ``unk_token_id``.

        Examples
        --------
        >>> tokenizer.convert_tokens_to_ids("ගෙ")
        10
        >>> tokenizer.convert_tokens_to_ids(["ගෙ", "ද"])
        [10, 11]
        """
        if not self.vocab_map:
            raise ValueError("Tokenizer not trained. Call train() or from_pretrained() first.")
        if isinstance(tokens, str):
            return self.vocab_map.get(tokens, self.unk_token_id or 1)
        return [self.vocab_map.get(t, self.unk_token_id or 1) for t in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        """
        Convert integer token ID(s) to their corresponding string token(s).

        Parameters
        ----------
        ids : int or list of int
            A single token ID or a list of token IDs.

        Returns
        -------
        str or list of str
            The corresponding token string(s).  Unknown IDs return
            ``unk_token``.

        Examples
        --------
        >>> tokenizer.convert_ids_to_tokens(10)
        'ගෙ'
        >>> tokenizer.convert_ids_to_tokens([10, 11])
        ['ගෙ', 'ද']
        """
        if not self.token_id_to_token_map:
            raise ValueError("Tokenizer not trained. Call train() or from_pretrained() first.")
        if isinstance(ids, int):
            return self.token_id_to_token_map.get(ids, self.unk_token)
        return [self.token_id_to_token_map.get(i, self.unk_token) for i in ids]

    def get_vocab(self) -> Dict[str, int]:
        """
        Return a copy of the full vocabulary mapping.

        Returns
        -------
        dict of str to int
            Mapping from token string to integer ID.

        Raises
        ------
        ValueError
            If the tokenizer has not been trained or loaded.
        """
        if not self.vocab_map:
            raise ValueError("Tokenizer not trained. Call train() or from_pretrained() first.")
        return dict(self.vocab_map)

    @property
    def vocab_size(self) -> int:
        """
        Total number of tokens in the vocabulary.

        Returns
        -------
        int
            Number of tokens including special tokens.
        """
        return len(self.vocab_map) if self.vocab_map else 0

    @property
    def all_special_tokens(self) -> List[str]:
        """
        List of all special token strings.

        Returns
        -------
        list of str
            ``[pad_token, unk_token, eos_token, bos_token]``
        """
        return list(self.special_tokens)

    # ------------------------------------------------------------------
    # Batch encode convenience method (kept for backward compat)
    # ------------------------------------------------------------------

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        add_bos_token: bool = False,
        allowed_special_tokens: Optional[List[str]] = None,
        # Legacy
        truncate_and_pad: bool = False,
    ) -> BatchEncoding:
        """
        Encode a list of strings into a batched :class:`~sinlib.BatchEncoding`.

        Parameters
        ----------
        texts : list of str
            Texts to encode.
        padding : bool, optional
            Pad sequences to the same length.  Default ``False``.
        truncation : bool, optional
            Truncate sequences at ``max_length``.  Default ``False``.
        max_length : int, optional
            Length limit; falls back to ``model_max_length``.
        return_attention_mask : bool, optional
            Include ``"attention_mask"`` in output.  Default ``True``.
        add_bos_token : bool, optional
            Prepend BOS token.  Default ``False``.
        allowed_special_tokens : list of str, optional
            Special tokens to encode literally.
        truncate_and_pad : bool, optional
            Legacy alias for ``padding=True, truncation=True``.

        Returns
        -------
        BatchEncoding
            A ``BatchEncoding`` where ``.input_ids`` is a list-of-lists.
        """
        if truncate_and_pad:
            padding = True
            truncation = True
        return self(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            add_bos_token=add_bos_token,
            allowed_special_tokens=allowed_special_tokens or [],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        text_list: List[str],
        memory_efficient: bool = True,
        chunk_size: int = 1000,
    ) -> None:
        """
        Build the vocabulary by training on a list of text strings.

        Parameters
        ----------
        text_list : list of str
            Training corpus.  Each element is one text sample.
        memory_efficient : bool, optional
            Process the corpus in ``chunk_size`` batches to reduce peak memory
            usage.  Default ``True``.
        chunk_size : int, optional
            Number of texts processed per batch when ``memory_efficient=True``.
            Default ``1000``.

        Raises
        ------
        ValueError
            If ``text_list`` is empty.

        Examples
        --------
        >>> tokenizer = Tokenizer(model_max_length=128)
        >>> tokenizer.train(["මම ගෙදර ගියා", "ඔහු පාසලට ගියා"])
        >>> tokenizer.vocab_size
        20
        """
        if not text_list:
            raise ValueError("Empty text list provided for training.")

        if memory_efficient:
            self._train_memory_efficient(text_list, chunk_size)
        else:
            self._train_standard(text_list)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, save_path: str) -> None:
        """
        Save vocabulary and configuration to a directory.

        Creates ``vocab.json`` and ``config.json`` in ``save_path``.  The
        directory is created if it does not exist.

        Parameters
        ----------
        save_path : str
            Directory path where the tokenizer files will be written.

        Raises
        ------
        IOError
            If writing the files fails.

        Examples
        --------
        >>> tokenizer.save_pretrained("./my_tokenizer/")
        """
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            # Legacy keys so old code that reads config.json still works
            "unknown_token": self.unk_token,
            "end_of_text_token": self.eos_token,
            "unk_token_id": self.unk_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            # Legacy IDs
            "unknown_token_id": self.unk_token_id,
            "end_of_text_token_id": self.eos_token_id,
            "model_max_length": self.model_max_length,
            "max_length": self.model_max_length,
        }

        try:
            with open(path / "vocab.json", "w", encoding="utf-8") as f:
                json.dump(self.vocab_map, f, ensure_ascii=False, indent=4)
            with open(path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except OSError as exc:
            raise IOError(f"Error saving tokenizer: {exc}") from exc

    # Legacy alias
    def save_tokenizer(self, save_path: str) -> None:
        """
        Alias for :meth:`save_pretrained`.

        .. deprecated::
            Use :meth:`save_pretrained` instead.
        """
        warnings.warn(
            "save_tokenizer() is deprecated. Use save_pretrained() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.save_pretrained(save_path)

    # Legacy load method kept for backward compat
    def load_from_pretrained(
        self,
        file_path: Optional[str] = None,
        load_default_tokenizer: bool = True,
    ) -> None:
        """
        Load tokenizer from files (legacy method).

        .. deprecated::
            Use the class method :meth:`from_pretrained` instead::

                tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

        Parameters
        ----------
        file_path : str, optional
            Path to a local directory containing ``vocab.json`` and
            ``config.json``.
        load_default_tokenizer : bool, optional
            When ``True`` (default), ignores ``file_path`` and downloads the
            default tokenizer from HuggingFace Hub.

        Raises
        ------
        ValueError
            If both or neither of ``file_path`` / ``load_default_tokenizer``
            are supplied, or if the path does not exist.
        """
        warnings.warn(
            "load_from_pretrained() is deprecated. "
            "Use Tokenizer.from_pretrained('Ransaka/sinlib') instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if load_default_tokenizer and file_path is not None:
            raise ValueError(
                "Provide either file_path or load_default_tokenizer=True, not both."
            )

        if load_default_tokenizer:
            self._load_default_tokenizer()
            return

        if file_path is None:
            raise ValueError("Provide file_path or set load_default_tokenizer=True.")

        self._load_from_local(Path(file_path))

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pad_or_truncate(
        sequence: List[int], max_length: int, padding_value: int
    ) -> List[int]:
        """
        Pad or truncate a sequence to a fixed length.

        Parameters
        ----------
        sequence : list of int
            The input sequence.
        max_length : int
            Target length.
        padding_value : int
            Value used for padding when the sequence is shorter than
            ``max_length``.

        Returns
        -------
        list of int
            Sequence of exactly ``max_length`` elements.
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence + [padding_value] * (max_length - len(sequence))

    # ------------------------------------------------------------------
    # Private implementation helpers
    # ------------------------------------------------------------------

    def _encode_impl(
        self,
        text: str,
        padding: bool,
        truncation: bool,
        max_length: Optional[int],
        return_attention_mask: bool,
        add_bos_token: bool,
        allowed_special_tokens: List[str],
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """Encode a single string; returns a BatchEncoding."""
        if not self.vocab_map:
            raise ValueError(
                "Tokenizer not trained. Call train() or from_pretrained() first."
            )

        token_ids: List[int] = []

        if add_bos_token and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        eos_marker = self.eos_token
        parts = text.split(eos_marker)
        is_multi_part = len(parts) > 1

        for part in parts:
            tokens = process_text(part)
            for token in tokens:
                if token in self.special_tokens:
                    if token in allowed_special_tokens:
                        token_ids.append(self.vocab_map[token])
                else:
                    token_ids.append(
                        self.vocab_map.get(token, self.unk_token_id or 1)
                    )
            if is_multi_part and self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)

        effective_max = max_length
        should_pad = padding and effective_max is not None
        should_truncate = truncation and effective_max is not None

        if should_truncate and effective_max is not None:
            token_ids = token_ids[:effective_max]

        if should_pad and effective_max is not None and self.pad_token_id is not None:
            token_ids = self.pad_or_truncate(
                token_ids, effective_max, self.pad_token_id
            )

        if return_attention_mask:
            attn_mask = [
                0 if (self.pad_token_id is not None and tid == self.pad_token_id) else 1
                for tid in token_ids
            ]
            return BatchEncoding(
                {"input_ids": token_ids, "attention_mask": attn_mask},
                tensor_type=return_tensors,
            )

        return BatchEncoding({"input_ids": token_ids}, tensor_type=return_tensors)

    def _batch_encode_impl(
        self,
        texts: List[str],
        padding: bool,
        truncation: bool,
        max_length: Optional[int],
        return_attention_mask: bool,
        add_bos_token: bool,
        allowed_special_tokens: List[str],
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """Encode a list of strings; returns a BatchEncoding with list-of-lists."""
        if not self.vocab_map:
            raise ValueError(
                "Tokenizer not trained. Call train() or from_pretrained() first."
            )

        all_ids: List[List[int]] = []
        for text in texts:
            enc = self._encode_impl(
                text,
                padding=False,  # handle padding below after finding max len
                truncation=truncation,
                max_length=max_length,
                return_attention_mask=False,
                add_bos_token=add_bos_token,
                allowed_special_tokens=allowed_special_tokens,
            )
            all_ids.append(enc.input_ids)

        # Determine target length for padding
        if padding and max_length is None and self.model_max_length is None:
            # Pad to the longest sequence in this batch
            target_len: Optional[int] = max(len(ids) for ids in all_ids) if all_ids else 0
        else:
            target_len = max_length or self.model_max_length

        if padding and target_len is not None and self.pad_token_id is not None:
            all_ids = [
                self.pad_or_truncate(ids, target_len, self.pad_token_id)
                for ids in all_ids
            ]

        if return_attention_mask and self.pad_token_id is not None:
            all_masks: List[List[int]] = [
                [0 if tid == self.pad_token_id else 1 for tid in ids]
                for ids in all_ids
            ]
            return BatchEncoding(
                {"input_ids": all_ids, "attention_mask": all_masks},
                tensor_type=return_tensors,
            )

        return BatchEncoding({"input_ids": all_ids}, tensor_type=return_tensors)

    def _load_default_tokenizer(self) -> None:
        """Download and apply the default pretrained tokenizer from HF Hub."""
        self.vocab_map = load_default_vocab_map()
        config = load_default_config()
        self._apply_config(config)
        assert self.vocab_map is not None
        self.token_id_to_token_map = {v: k for k, v in self.vocab_map.items()}
        self._update_special_token_ids()

    def _load_from_local(self, path: Path) -> None:
        """Load vocabulary and config from a local directory."""
        vocab_file = path / "vocab.json"
        config_file = path / "config.json"

        if not vocab_file.exists():
            raise ValueError(f"vocab.json not found in {path}")
        if not config_file.exists():
            raise ValueError(f"config.json not found in {path}")

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab_map = json.load(f)
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        self._apply_config(config)
        assert self.vocab_map is not None
        self.token_id_to_token_map = {v: k for k, v in self.vocab_map.items()}
        self._update_special_token_ids()

    def _apply_config(self, config: dict) -> None:
        """Apply a config dict to this instance, supporting both old and new key names."""
        # Prefer new-style keys; fall back to legacy keys
        self.unk_token = config.get("unk_token", config.get("unknown_token", self.unk_token))
        self.pad_token = config.get("pad_token", self.pad_token)
        self.eos_token = config.get("eos_token", config.get("end_of_text_token", self.eos_token))
        self.bos_token = config.get("bos_token", self.bos_token)

        max_len = config.get("model_max_length", config.get("max_length"))
        if max_len is not None and self.model_max_length is None:
            self.model_max_length = max_len
            self.max_length = max_len

        # Legacy aliases
        self.unknown_token = self.unk_token
        self.end_of_text_token = self.eos_token

        self.special_tokens = [
            self.pad_token, self.unk_token, self.eos_token, self.bos_token
        ]

    def _update_special_token_ids(self) -> None:
        """Populate all special-token ID fields from the vocabulary."""
        if not self.vocab_map:
            return
        self.pad_token_id = self.vocab_map.get(self.pad_token)
        self.unk_token_id = self.vocab_map.get(self.unk_token)
        self.eos_token_id = self.vocab_map.get(self.eos_token)
        self.bos_token_id = self.vocab_map.get(self.bos_token)
        # Legacy aliases
        self.unknown_token_id = self.unk_token_id
        self.end_of_text_token_id = self.eos_token_id

    def _build_vocab_from_chars(self, unique_chars: Set[str]) -> None:
        """Assign integer IDs to all tokens, with special tokens at fixed low IDs."""
        self.unique_chars = unique_chars
        self.vocab_map = {}

        # Special tokens occupy IDs 0-3 in this fixed order
        for idx, token in enumerate(
            [self.pad_token, self.unk_token, self.eos_token, self.bos_token]
        ):
            self.vocab_map[token] = idx

        for char in unique_chars:
            if char not in self.vocab_map:
                self.vocab_map[char] = len(self.vocab_map)

        self.token_id_to_token_map = {v: k for k, v in self.vocab_map.items()}
        self._update_special_token_ids()

    def _train_memory_efficient(
        self, text_list: List[str], chunk_size: int
    ) -> None:
        """Train in memory-efficient mode by processing chunks."""
        unique_chars: Set[str] = set()
        total_chunks = (len(text_list) + chunk_size - 1) // chunk_size

        for i in tqdm(
            range(0, len(text_list), chunk_size),
            total=total_chunks,
            desc="Training tokenizer",
        ):
            chunk = text_list[i: i + chunk_size]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_text, chunk))
            for sublist in results:
                unique_chars.update(sublist)

        self._build_vocab_from_chars(unique_chars)

    def _train_standard(self, text_list: List[str]) -> None:
        """Train using a single parallel pass over all texts."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_text, text_list))
        all_chars = [char for sublist in results for char in sublist]
        self._build_vocab_from_chars(set(all_chars))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        trained = self.vocab_map is not None
        return (
            f"Tokenizer("
            f"vocab_size={self.vocab_size}, "
            f"model_max_length={self.model_max_length}, "
            f"trained={trained})"
        )
