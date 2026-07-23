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
from .utils.preprocessing import load_char_mapper, remove_non_printable, process_text


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
        self.tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

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
        if not text:
            return ""

        # Fallback mappings for diacritics/modifiers
        char_map = dict(self.char_mapper)
        char_map.setdefault('ං', 'n')
        char_map.setdefault('ඃ', 'h')

        tokens = process_text(text)
        res = []
        for tok in tokens:
            clean = tok.strip()
            roman = char_map.get(clean, clean)
            if tok.endswith(' ') and not roman.endswith(' '):
                roman += ' '
            res.append(roman)

        return "".join(res)
