"""
Module for transliterating text between Sinhala and Roman scripts.

This module provides the Transliterator class which handles the conversion
of text from one script to another using a pre-trained model.
"""
from typing import List

from sinlib.utils.model_utils import load_transliterator_model, inference
from sinlib.utils.dataset_utils import load_tokenizer


class Transliterator:
    """
    A class for transliterating text between scripts using a pre-trained model.
    
    This class loads a pre-trained transliteration model and provides methods
    to convert text from one script to another (typically between Sinhala and Roman).
    
    Attributes:
        model: The pre-trained transliteration model
        tokenizer: The tokenizer used for encoding/decoding text
    """
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None) -> None:
        """
        Initialize the Transliterator with a model and tokenizer.
        
        Args:
            model_path: Optional path to a custom model file
            tokenizer_path: Optional path to a custom tokenizer file
        """
        self.model = load_transliterator_model()
        self.tokenizer = load_tokenizer()
    
    def transliterate(self, text: str) -> str:
        """
        Transliterate the input text.
        
        Args:
            text: The input text to transliterate
            
        Returns:
            The transliterated text
            
        Examples:
            >>> transliterator = Transliterator()
            >>> transliterator.transliterate("මම ගෙදර ගියා")
            "mama gedara giya"
        """
        if not text or not isinstance(text, str):
            return ""
            
        word_list = text.split()
        transliterated_text = [
            inference(self.model, self.tokenizer, word) for word in word_list
        ]
        return " ".join(transliterated_text).strip()
    
    def __call__(self, text: str) -> str:
        """
        Make the class callable for easy transliteration.
        
        Args:
            text: The input text to transliterate
            
        Returns:
            The transliterated text
        """
        return self.transliterate(text)
    
    def batch_transliterate(self, texts: List[str]) -> List[str]:
        """
        Transliterate a batch of texts.
        
        Args:
            texts: A list of input texts to transliterate
            
        Returns:
            A list of transliterated texts
        """
        return [self.transliterate(text) for text in texts]
