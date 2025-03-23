from difflib import get_close_matches
from typing import List, Dict, Union, Any
import warnings
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
    
    def __init__(self):
        """
        Initialize the TypoDetector with a dictionary and n-gram probabilities.
        
        Args:
            dictionary_list: List of valid words.
            ngram_probs: Dictionary mapping n-gram keys to probabilities.
            tokenizer: Function to tokenize words. If None, identity function is used.
        """
        self._dictionary = self._load_dictionary()
        self._ngram_probs = self._load_ngram_probs()
        self._tokenizer = self._load_tokenizer()
    
    def _load_dictionary(self) -> List[str]:
        """
        Load the dictionary from a file.

        Returns:
            List of valid words.
        """
        dictionary_path = download_hub_file(Filenames.DICTIONARY.value)
        return list(np.load(dictionary_path))

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
    
    def check_spelling(self, word: str) -> str:
        """
        Check if a word is spelled correctly and suggest corrections if not.
        
        Args:
            word: The word to check.
            
        Returns:
            A message indicating if the word is valid, a typo, or unusual.
        """
        if word in self._dictionary:
            return word
        
        prob = self.word_ngram_probability(word)
        
        if prob < 1e-8:
            suggestions = self.suggest_correction(word)
            return suggestions
        warnings.warn(f"'{word}' is unusual but may not be a typo", UserWarning)
        return word