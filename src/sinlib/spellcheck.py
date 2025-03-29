from difflib import get_close_matches
from typing import List, Dict, Union, Any
import warnings
from functools import lru_cache
from sinlib.tokenizer import Tokenizer
from sinlib.utils.preprocessing import download_hub_file, Filenames
import numpy as np

class TypoDetector:
    """
    A class for detecting and correcting typos in words using n-gram probabilities.
    
    Attributes:
        _dictionary (List[str]): List of valid words.
        _ngram_probs (Dict[int, float]): Dictionary of n-gram probabilities.
    """
    
    def __init__(self, cache_size: int = 1000, threshold: float = 1e-8, lazy_loading: bool = False):
        """
        Initialize the TypoDetector with configurable caching and loading options.
        
        Args:
            cache_size: Maximum number of entries to cache for frequent operations
            threshold: Probability threshold for considering words as typos
            lazy_loading: Delay resource loading until first use
        """
        self._cache_size = cache_size
        self._threshold = threshold
        self._lazy_loading = lazy_loading
        
        if not lazy_loading:
            self._dictionary = self._load_dictionary()
            self._ngram_probs = self._load_ngram_probs()
            self._tokenizer = self._load_tokenizer()
            # Apply caching to core methods
            self.word_ngram_probability = lru_cache(maxsize=cache_size)(self.word_ngram_probability)
            self.suggest_correction = lru_cache(maxsize=cache_size)(self.suggest_correction)
            self.__call__ = lru_cache(maxsize=cache_size)(self.__call__)
    
    def _load_dictionary(self) -> set:
        """
        Load the dictionary as a set for O(1) lookups.

        Returns:
            Set of valid words.
        """
        dictionary_path = download_hub_file(Filenames.DICTIONARY.value)
        return set(np.load(dictionary_path).tolist())

    def _load_ngram_probs(self) -> Dict[int, float]:
        """
        Load the n-gram probabilities from a file.
        Returns:
            Dictionary mapping n-gram keys to probabilities.
        """
        ngram_probs_path = download_hub_file(Filenames.NGRAM_PROBS.value)
        loaded_data = np.load(ngram_probs_path, allow_pickle=True)
        return loaded_data.item() if hasattr(loaded_data, 'item') else loaded_data

    def _load_tokenizer(self) -> Tokenizer:
        """
        Load the tokenizer from a file.
        Returns:
            Tokenizer object.
        """
        _tokenizer = Tokenizer(max_length=10)
        _tokenizer.load_from_pretrained(load_default_tokenizer=True)
        return _tokenizer
    
    @property
    def dictionary(self) -> str:
        """Return a description of the dictionary."""
        return f"Dictionary containing {len(self._dictionary)} words. Use .get_dictionary() to access the full list."
    
    def get_dictionary(self) -> List[str]:
        """Return the full dictionary list."""
        return self._dictionary
    
    @property
    def ngram_probs(self) -> str:
        """Return a description of the n-gram probabilities."""
        return f"N-gram probability dictionary with {len(self._ngram_probs)} entries. Use .get_ngram_probs() to access the full dictionary."
    
    def get_ngram_probs(self) -> Dict[int, float]:
        """Return the full n-gram probabilities dictionary."""
        return self._ngram_probs
    
    @lru_cache(maxsize=1000)
    def word_ngram_probability(self, word: str, n: int = 2) -> float:
        """
        Calculate the probability of a word based on its n-grams.
        
        Args:
            word: The word to calculate probability for.
            n: Size of n-grams to use.
            
        Returns:
            Probability score for the word.
        """
        word = self._tokenizer(word, truncate_and_pad=False)
        prob = 1.0
        for i in range(len(word) - n + 1):
            ngram = word[i:i+n]
            ngram_key = map(str, ngram)
            ngram_key = "".join(ngram_key)
            prob *= self._ngram_probs.get(int(ngram_key), 1e-9)  # Small value for unseen n-grams
        return prob
    
    def suggest_correction(self, word: str, n: int = 3) -> List[str]:
        """
        Find closest valid words using edit distance.
        
        Args:
            word: The word to find corrections for.
            n: Maximum number of suggestions to return.
            
        Returns:
            List of suggested corrections.
        """
        matches = get_close_matches(word, self._dictionary, n=n, cutoff=0.7)
        return matches if matches else ["No suggestion"]
    
    def __call__(self, text: str) -> str:
        """
        Check text for spelling errors and return corrected sentence.
        
        Args:
            text: The sentence to check.
            
        Returns:
            Corrected sentence.
        """
        corrected = []
        words = text.split() if isinstance(text, str) else [str(text)]
        
        for w in words:
            try:
                # Convert dictionary to set for O(1) lookups
                if w in set(self._dictionary):
                    corrected.append(w)
                    continue
                
                prob = self.word_ngram_probability(w)
                
                if prob < 1e-8:
                    suggestions = self.suggest_correction(w)
                    corrected.append(suggestions[0] if suggestions else w)
                else:
                    warnings.warn(f"'{w}' is unusual but may not be a typo", UserWarning)
                    corrected.append(w)
            except Exception as e:
                warnings.warn(f"Error processing word '{w}': {str(e)}")
                corrected.append(w)
        
        return ' '.join(corrected)