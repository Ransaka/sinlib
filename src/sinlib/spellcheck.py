"""
Spell-checking module for Sinhala text.

Uses n-gram language model probabilities combined with edit-distance-based
candidate generation to detect and correct misspelled Sinhala words.
"""

from __future__ import annotations

import warnings
from difflib import get_close_matches
from functools import lru_cache
from typing import Dict, List, Optional, Set

import numpy as np

from sinlib.tokenizer import Tokenizer
from sinlib.utils.preprocessing import download_hub_file, Filenames

# Default HF Hub repo
_DEFAULT_HF_REPO = "Ransaka/sinlib"


class TypoDetector:
    """
    Detect and correct Sinhala spelling errors using n-gram language models.

    For each word in a sentence the detector checks whether the word is in the
    known dictionary.  If not, it estimates the word's likelihood via bigram
    probabilities.  Words below the ``threshold`` probability are replaced by
    the closest dictionary match returned by :meth:`suggest_correction`.

    Parameters
    ----------
    cache_size : int, optional
        LRU-cache size for :meth:`word_ngram_probability` and
        :meth:`suggest_correction`.  Default ``1000``.
    threshold : float, optional
        Minimum n-gram probability for a word to be considered valid.
        Default ``1e-8``.
    lazy_loading : bool, optional
        When ``True``, defer loading dictionary/model data until the first
        call.  Default ``False``.

    Examples
    --------
    Load the default detector from HuggingFace Hub:

    >>> from sinlib import TypoDetector
    >>> detector = TypoDetector.from_pretrained("Ransaka/sinlib")
    >>> detector("අඩිරාජයාගේ")
    'අධිරාජයාගේ'

    Or initialise directly (downloads on construction):

    >>> detector = TypoDetector()
    >>> detector("අපකරියට ගිය")
    'අපකීර්තියට ගිය'
    """

    # ------------------------------------------------------------------
    # Class-method constructor (HF-style)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = _DEFAULT_HF_REPO,
        cache_size: int = 1000,
        threshold: float = 1e-8,
    ) -> "TypoDetector":
        """
        Load a :class:`TypoDetector` from HuggingFace Hub.

        Parameters
        ----------
        pretrained_model_name_or_path : str, optional
            HuggingFace Hub repo ID.  Currently only ``"Ransaka/sinlib"``
            (the default) is supported.
        cache_size : int, optional
            LRU-cache size.  Default ``1000``.
        threshold : float, optional
            N-gram probability threshold.  Default ``1e-8``.

        Returns
        -------
        TypoDetector
            A fully initialised detector ready for use.

        Raises
        ------
        ValueError
            If an unsupported repo ID is provided.

        Examples
        --------
        >>> detector = TypoDetector.from_pretrained("Ransaka/sinlib")
        >>> detector("සිංහල")
        'සිංහල'
        """
        if pretrained_model_name_or_path not in (_DEFAULT_HF_REPO, "sinlib"):
            raise ValueError(
                f"Repository '{pretrained_model_name_or_path}' is not supported. "
                f"Use '{_DEFAULT_HF_REPO}'."
            )
        return cls(cache_size=cache_size, threshold=threshold, lazy_loading=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        cache_size: int = 1000,
        threshold: float = 1e-8,
        lazy_loading: bool = False,
    ) -> None:
        self._cache_size = cache_size
        self._threshold = threshold
        self._lazy_loading = lazy_loading

        self._dictionary: Optional[Set[str]] = None
        self._ngram_probs: Optional[Dict] = None
        self._tokenizer: Optional[Tokenizer] = None

        if not lazy_loading:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load resources if they have not been loaded yet."""
        if self._dictionary is None:
            self._dictionary = self._load_dictionary()
        if self._ngram_probs is None:
            self._ngram_probs = self._load_ngram_probs()
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()

    # ------------------------------------------------------------------
    # Resource loaders
    # ------------------------------------------------------------------

    def _load_dictionary(self) -> Set[str]:
        """
        Load the Sinhala word dictionary from HuggingFace Hub.

        Returns
        -------
        set of str
            Set of known valid Sinhala words.
        """
        path = download_hub_file(Filenames.DICTIONARY.value)
        return set(np.load(path).tolist())

    def _load_ngram_probs(self) -> Dict:
        """
        Load bigram probability table from HuggingFace Hub.

        Returns
        -------
        dict
            Mapping from n-gram key strings to probability floats.
        """
        path = download_hub_file(Filenames.NGRAM_PROBS.value)
        loaded = np.load(path, allow_pickle=True)
        return loaded.item() if hasattr(loaded, "item") else dict(loaded)

    def _load_tokenizer(self) -> Tokenizer:
        """
        Load the default Sinhala tokenizer.

        Returns
        -------
        Tokenizer
            Pretrained tokenizer for character-level encoding.
        """
        return Tokenizer.from_pretrained(_DEFAULT_HF_REPO, model_max_length=10)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dictionary(self) -> str:
        """
        Human-readable summary of the loaded dictionary.

        Returns
        -------
        str
            Description including the word count.
        """
        self._ensure_loaded()
        return (
            f"Dictionary containing {len(self._dictionary)} words. "  # type: ignore[arg-type]
            "Use .get_dictionary() to access the full list."
        )

    @property
    def ngram_probs(self) -> str:
        """
        Human-readable summary of the loaded n-gram probability table.

        Returns
        -------
        str
            Description including the entry count.
        """
        self._ensure_loaded()
        return (
            f"N-gram probability dictionary with {len(self._ngram_probs)} entries. "  # type: ignore[arg-type]
            "Use .get_ngram_probs() to access the full dictionary."
        )

    def get_dictionary(self) -> Set:
        """
        Return the full word dictionary.

        Returns
        -------
        set of str
            All known valid Sinhala words.
        """
        self._ensure_loaded()
        return self._dictionary  # type: ignore[return-value]

    def get_ngram_probs(self) -> Dict:
        """
        Return the full n-gram probability table.

        Returns
        -------
        dict
            Bigram probability mapping.
        """
        self._ensure_loaded()
        return self._ngram_probs  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @lru_cache(maxsize=1000)
    def word_ngram_probability(self, word: str, n: int = 2) -> float:
        """
        Estimate the probability of a word using character-level n-grams.

        Parameters
        ----------
        word : str
            Input word to score.
        n : int, optional
            N-gram size.  Default ``2`` (bigrams).

        Returns
        -------
        float
            Product of all n-gram probabilities.  Unseen n-grams contribute
            a small smoothing value (``1e-9``).

        Examples
        --------
        >>> detector.word_ngram_probability("සිංහල")
        3.2e-05
        """
        self._ensure_loaded()
        token_ids = self._tokenizer.encode(word)  # type: ignore[union-attr]
        prob = 1.0
        for i in range(len(token_ids) - n + 1):
            ngram_key = "".join(map(str, token_ids[i: i + n]))
            prob *= self._ngram_probs.get(int(ngram_key), 1e-9)  # type: ignore[union-attr]
        return prob

    @lru_cache(maxsize=1000)
    def suggest_correction(self, word: str, n: int = 3) -> List[str]:
        """
        Return the closest dictionary matches for a misspelled word.

        Parameters
        ----------
        word : str
            Misspelled word.
        n : int, optional
            Maximum number of suggestions.  Default ``3``.

        Returns
        -------
        list of str
            Up to ``n`` candidate corrections ordered by similarity.
            Returns ``["No suggestion"]`` when no close match is found.

        Examples
        --------
        >>> detector.suggest_correction("අඩිරාජ")
        ['අධිරාජ']
        """
        self._ensure_loaded()
        matches = get_close_matches(
            word, self._dictionary, n=n, cutoff=0.7  # type: ignore[arg-type]
        )
        return list(matches) if matches else ["No suggestion"]

    def __call__(self, text: str) -> str:
        """
        Check a sentence for spelling errors and return the corrected version.

        Parameters
        ----------
        text : str
            Input sentence to check.

        Returns
        -------
        str
            Sentence with detected typos replaced by the top suggestion.

        Examples
        --------
        >>> detector("අපකරියට ගිය")
        'අපකීර්තියට ගිය'
        """
        self._ensure_loaded()
        corrected: List[str] = []
        words = text.split() if isinstance(text, str) else [str(text)]

        for word in words:
            try:
                if word in self._dictionary:  # type: ignore[operator]
                    corrected.append(word)
                    continue

                prob = self.word_ngram_probability(word)

                if prob < self._threshold:
                    suggestions = self.suggest_correction(word)
                    corrected.append(
                        suggestions[0] if suggestions[0] != "No suggestion" else word
                    )
                else:
                    warnings.warn(
                        f"'{word}' is unusual but may not be a typo.",
                        UserWarning,
                        stacklevel=2,
                    )
                    corrected.append(word)

            except Exception as exc:
                warnings.warn(
                    f"Error processing word '{word}': {exc}",
                    stacklevel=2,
                )
                corrected.append(word)

        return " ".join(corrected)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._dictionary is not None
        n_words = len(self._dictionary) if self._dictionary is not None else 0
        return (
            f"TypoDetector("
            f"loaded={loaded}, "
            f"dictionary_size={n_words}, "
            f"threshold={self._threshold})"
        )
