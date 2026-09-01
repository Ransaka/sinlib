"""
Phonological Unit (Akshara) Tokenizer for the Character Channel.
Wraps sinlib phonological segmentation with specialized special token handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from sinlib.utils.preprocessing import normalize_sinhala, process_text


class SinhalaCharTokenizer:
    """
    Phonological Akshara Tokenizer for Sinhala-CharBERT Character Channel.
    Maps Sinhala text into phonological unit sequences with reserved special tokens.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    MASK_TOKEN = "<mask>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]

    def __init__(self, vocab_map: Optional[Dict[str, int]] = None):
        if vocab_map is not None:
            self.vocab_map = dict(vocab_map)
        else:
            self.vocab_map = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}

        self.inv_vocab_map = {v: k for k, v in self.vocab_map.items()}
        self.pad_token_id = self.vocab_map[self.PAD_TOKEN]
        self.unk_token_id = self.vocab_map[self.UNK_TOKEN]
        self.bos_token_id = self.vocab_map[self.BOS_TOKEN]
        self.eos_token_id = self.vocab_map[self.EOS_TOKEN]
        self.mask_token_id = self.vocab_map[self.MASK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_map)

    def train_on_corpus(self, texts: List[str]) -> None:
        """Extracts all unique phonological units from texts and builds vocabulary."""
        unique_units: Set[str] = set()
        for t in texts:
            if not t:
                continue
            norm_t = normalize_sinhala(t)
            units = process_text(norm_t)
            unique_units.update(units)

        for unit in sorted(unique_units):
            if unit not in self.vocab_map:
                self.vocab_map[unit] = len(self.vocab_map)

        self.inv_vocab_map = {v: k for k, v in self.vocab_map.items()}

    def tokenize(self, text: str) -> List[str]:
        """Segments text into phonological Akshara units."""
        norm_text = normalize_sinhala(text)
        return process_text(norm_text)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
    ) -> List[int]:
        """Encodes text into a list of phonological unit integer IDs."""
        units = self.tokenize(text)
        ids = [self.vocab_map.get(u, self.unk_token_id) for u in units]

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif pad_to_max_length and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes phonological unit integer IDs back to string."""
        units = []
        special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id}
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            units.append(self.inv_vocab_map.get(tid, self.UNK_TOKEN))
        return "".join(units)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves character tokenizer vocabulary to a JSON file."""
        import json
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab_map, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SinhalaCharTokenizer":
        """Loads character tokenizer vocabulary from a JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab_map=vocab)

    def load_vocab(self, filepath: Union[str, Path]) -> "SinhalaCharTokenizer":
        """Loads character tokenizer vocabulary in-place from a JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab_map = json.load(f)
        self.inv_vocab_map = {v: k for k, v in self.vocab_map.items()}
        self.pad_token_id = self.vocab_map.get(self.PAD_TOKEN, 0)
        self.unk_token_id = self.vocab_map.get(self.UNK_TOKEN, 1)
        self.bos_token_id = self.vocab_map.get(self.BOS_TOKEN, 2)
        self.eos_token_id = self.vocab_map.get(self.EOS_TOKEN, 3)
        self.mask_token_id = self.vocab_map.get(self.MASK_TOKEN, 4)
        return self

