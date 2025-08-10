"""
Module for romanizing Sinhala text.

This module provides functionality to convert Sinhala text to its romanized form
using character mapping and tokenization.
"""
from typing import Dict, List, Optional, Set, Union

import numpy as np
from numpy.typing import NDArray

from .tokenizer import Tokenizer
from .utils.chars import ALL_SINHALA_CHARACTERS, NUMBERS_AND_PUNCTUATION
from .utils.preprocessing import load_char_mapper, remove_non_printable


class Romanizer:
    """
    A class for converting Sinhala text to Roman characters.
    
    This class provides functionality to convert Sinhala text to its romanized
    form while preserving non-Sinhala characters and maintaining word boundaries.
    
    Attributes:
        char_mapper: Dictionary mapping Sinhala characters to their Roman equivalents
        tokenizer: Tokenizer instance for processing Sinhala text
    """

    def __init__(
        self, 
        char_mapper_fp: Optional[str] = None, 
        tokenizer_path: Optional[str] = None
    ) -> None:
        """
        Initialize the Romanizer with character mappings and tokenizer.
        
        Args:
            char_mapper_fp: Path to character mapping file
            tokenizer_path: Path to tokenizer vocabulary file
        """
        self.char_mapper = load_char_mapper()
        self.tokenizer = Tokenizer(max_length=None)
        self.tokenizer.load_from_pretrained(file_path=None, load_default_tokenizer=True)

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Convert input text to romanized form.
        
        Args:
            text: Input text or list of texts to romanize
            
        Returns:
            Romanized version of the input text
        """
        if isinstance(text, list):
            return [self.__romanize(t) for t in text]
        return self.__romanize(text)

    def __romanize(self, text: str) -> str:
        """
        Convert a single text to its romanized form.
        
        Args:
            text: Input text to romanize
            
        Returns:
            Romanized version of the input text
        """
        text = remove_non_printable(text)
        chars: NDArray = np.array(list(text))
        
        # Create mask for Sinhala characters and allowed punctuation
        sinhala_mask = [
            char in ALL_SINHALA_CHARACTERS + list(NUMBERS_AND_PUNCTUATION) + [" "]
            for char in chars
        ]
        
        # Extract Sinhala text
        sinhala_text = "".join(chars[sinhala_mask]).strip()
        
        # Tokenize and decode
        encodings = self.tokenizer(sinhala_text, truncate_and_pad=False)
        decoded_chars = [
            self.tokenizer.token_id_to_token_map[c] for c in encodings
        ]
        
        # Convert to Roman characters
        romanized_chars = [
            self.char_mapper.get(ch, ch if ch in NUMBERS_AND_PUNCTUATION.union({" "}) else '')
            for ch in decoded_chars
        ]
        romanized_sinhala = "".join(romanized_chars)
        
        # Create word mapping and apply to full text
        word_mapping = dict(zip(sinhala_text.split(), romanized_sinhala.split()))
        romanized_words = [word_mapping.get(word, word) for word in text.split()]
        
        return " ".join(romanized_words)
