"""
Sinhala text tokenization module.

This module provides a character-level tokenizer specifically designed for Sinhala text,
with support for special tokens, memory-efficient training, and vocabulary management.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import concurrent.futures

from tqdm import tqdm

from .utils.preprocessing import process_text, load_default_vocab_map, load_default_config


class Tokenizer:
    def __init__(
        self, 
        max_length: int,
        unknown_token: str = "<|unk|>",
        pad_token: str = "<|pad|>",
        end_of_text_token: str = "<|end_of_text|>"
    ) -> None:
        """Initialize the tokenizer with specified parameters."""
        # Special tokens
        self.unknown_token: str = unknown_token
        self.pad_token: str = pad_token
        self.end_of_text_token: str = end_of_text_token
        self.special_tokens: List[str] = [self.unknown_token, self.pad_token]
        
        # Configuration
        self.max_length: int = max_length
        
        # Token mappings
        self.vocab_map: Optional[Dict[str, int]] = None
        self.token_id_to_token_map: Optional[Dict[int, str]] = None
        
        # Token IDs
        self.unknown_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.end_of_text_token_id: Optional[int] = None
        
        # Training state
        self.tokenized_chars: List[str] = []
        self.unique_chars: Set[str] = set()

    def __encode(
        self,
        text: str,
        truncate_and_pad: bool,
        allowed_special_tokens: List[str] = []
    ) -> List[int]:
        """Encode text into token IDs."""
        if not self.vocab_map:
            raise ValueError("Tokenizer not trained. Call train() first.")

        allowed_token_ids = [self.vocab_map[tok] for tok in allowed_special_tokens]
        text_encodings: List[int] = []
        
        for part in text.split(self.end_of_text_token):
            processed_text = self.__process_text(part)
            
            for token in processed_text:
                if token in self.special_tokens:
                    if token in allowed_special_tokens:
                        text_encodings.append(self.vocab_map[token])
                else:
                    text_encodings.append(self.vocab_map.get(token, self.unknown_token_id))
            
            if len(text.split(self.end_of_text_token)) > 1:
                text_encodings.append(self.end_of_text_token_id)

        return (
            self.pad_or_truncate(text_encodings, self.max_length, self.pad_token_id)
            if truncate_and_pad
            else text_encodings
        )

    @staticmethod
    def pad_or_truncate(sequence: List[int], max_length: int, padding_value: int) -> List[int]:
        """Pad or truncate a sequence to specified length."""
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence + [padding_value] * (max_length - len(sequence))

    def __call__(
        self,
        text: str,
        truncate_and_pad: bool = False,
        allowed_special_tokens: List[str] = []
    ) -> List[int]:
        """Make the class callable for easy encoding."""
        return self.__encode(text, truncate_and_pad, allowed_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        if not self.token_id_to_token_map:
            raise ValueError("Tokenizer not trained. Call train() first.")

        special_token_ids = [self.vocab_map[tok] for tok in self.special_tokens]
        tokens = [
            token for token in ids
            if not skip_special_tokens or token not in special_token_ids
        ]
        
        return "".join(
            self.token_id_to_token_map.get(token, self.unknown_token)
            for token in tokens
        )

    def train(
        self,
        text_list: List[str],
        memory_efficient: bool = True,
        chunk_size: int = 1000
    ) -> None:
        """Train the tokenizer on a list of text strings."""
        if not text_list:
            raise ValueError("Empty text list provided for training")

        if memory_efficient:
            self.__train_character_level_tokenizer_memory_efficient(text_list, chunk_size)
        else:
            self.__train_character_level_tokenizer(text_list)

    def __len__(self) -> int:
        """Get the vocabulary size."""
        return len(self.vocab_map) if self.vocab_map else 0

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size as a property."""
        return len(self)

    @staticmethod
    def __process_text(text: str) -> List[str]:
        """Process text using utility function."""
        return process_text(text)

    def __load_default_tokenizer(self) -> None:
        """Load default tokenizer."""
        self.vocab_map = load_default_vocab_map()
        config = load_default_config()

        # Load configuration
        for key, value in config.items():
            setattr(self, key, value)
        
        self.token_id_to_token_map = {v: k for k, v in self.vocab_map.items()}
        self.__update_special_token_ids()

    def __train_character_level_tokenizer_memory_efficient(
        self,
        text_list: List[str],
        chunk_size: int
    ) -> None:
        """Train tokenizer in memory-efficient mode."""
        unique_chars: Set[str] = set()
        total_chunks = (len(text_list) + chunk_size - 1) // chunk_size

        for i in tqdm(range(0, len(text_list), chunk_size), total=total_chunks, desc="Training tokenizer"):
            chunk = text_list[i:i + chunk_size]
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(self.__process_text, chunk))
                
            for sublist in results:
                unique_chars.update(sublist)

        self.__build_vocab_from_chars(unique_chars)

    def __train_character_level_tokenizer(self, text_list: List[str]) -> None:
        """Train tokenizer using standard mode."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.__process_text, text_list))
            self.tokenized_chars = [char for sublist in results for char in sublist]
        
        self.__build_vocab_from_chars(set(self.tokenized_chars))

    def __build_vocab_from_chars(self, unique_chars: Set[str]) -> None:
        """Build vocabulary from unique characters."""
        self.unique_chars = unique_chars
        self.vocab_map = {char: idx for idx, char in enumerate(unique_chars)}
        
        # Add special tokens
        for token in [self.unknown_token, self.pad_token, self.end_of_text_token]:
            self.vocab_map[token] = len(self.vocab_map)
        
        # Set token IDs
        self.unknown_token_id = self.vocab_map[self.unknown_token]
        self.pad_token_id = self.vocab_map[self.pad_token]
        self.end_of_text_token_id = self.vocab_map[self.end_of_text_token]
        
        # Create reverse mapping
        self.token_id_to_token_map = {v: k for k, v in self.vocab_map.items()}

    def load_from_pretrained(self, file_path: str, load_default_tokenizer:bool = False) -> 'Tokenizer':
        """Load tokenizer from pretrained files."""

        if load_default_tokenizer and (file_path is not None):
            raise ValueError("Both file_path and load_default_tokenizer cannot be provide.")
        
        if load_default_tokenizer:
            self.__load_default_tokenizer()
            return self
        
        try:
            file_path = Path(file_path)
            if file_path.exists():
                with open(file_path / "vocab.json", "r", encoding="utf-8") as f:
                    self.vocab_map = json.load(f)
                with open(file_path / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # Load configuration
                for key, value in config.items():
                    setattr(self, key, value)
                
                            # Update mappings
                self.token_id_to_token_map = {v: k for k, v in self.vocab_map.items()}
                self.__update_special_token_ids()
                return self
            
            else:
                raise ValueError(
                    "File not found at the specified path. Loading default vocab map.",
                    UserWarning
                )
            
        except Exception as e:
            raise ValueError(f"Error loading pretrained tokenizer: {str(e)}")

    def __update_special_token_ids(self) -> None:
        """Update special token IDs from vocab map."""
        self.unknown_token_id = self.vocab_map[self.unknown_token]
        self.pad_token_id = self.vocab_map[self.pad_token]
        self.end_of_text_token_id = self.vocab_map[self.end_of_text_token]

    def save_tokenizer(self, save_path: str) -> None:
        """Save tokenizer configuration and vocabulary."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "unknown_token": self.unknown_token,
            "pad_token": self.pad_token,
            "unknown_token_id": self.unknown_token_id,
            "pad_token_id": self.pad_token_id,
            "max_length": self.max_length,
            "end_of_text_token": self.end_of_text_token,
            "end_of_text_token_id": self.end_of_text_token_id,
        }

        try:
            with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
                json.dump(self.vocab_map, f, ensure_ascii=False, indent=4)

            with open(save_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            raise IOError(f"Error saving tokenizer: {str(e)}")
