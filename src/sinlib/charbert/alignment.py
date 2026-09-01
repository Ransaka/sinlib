"""
Sequence Alignment Engine for mapping subword tokens to phonological character unit boundaries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

from sinlib.utils.preprocessing import normalize_sinhala, process_text
from sinlib.charbert.char_tokenization import SinhalaCharTokenizer


@dataclass
class AlignedSequence:
    """Represents a dual-channel aligned text sequence."""
    text: str
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]
    char_units: List[str]
    char_input_ids: List[int]
    char_attention_mask: List[int]
    start_char_idx: List[int]
    end_char_idx: List[int]
    subword_offsets: List[Tuple[int, int]] = field(default_factory=list)


class SequenceAlignmentEngine:
    """
    Computes exact bidirectional boundary mappings between Subword Tokens (m)
    and Phonological Units / Aksharas (N).
    """

    def __init__(
        self,
        subword_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer, Any],
        char_tokenizer: SinhalaCharTokenizer,
        max_subword_length: int = 256,
        max_char_length: int = 512,
    ):
        self.subword_tokenizer = subword_tokenizer
        self.char_tokenizer = char_tokenizer
        self.max_subword_length = max_subword_length
        self.max_char_length = max_char_length

    def _get_phonological_spans(self, text: str, units: List[str]) -> List[Tuple[int, int]]:
        """Calculates character string offsets [start_char, end_char) for each phonological unit in raw text."""
        spans: List[Tuple[int, int]] = []
        curr_idx = 0
        for u in units:
            # Locate unit starting from current offset
            found_idx = text.find(u, curr_idx)
            if found_idx == -1:
                # Fallback if normalization shifted index
                found_idx = curr_idx
            end_idx = found_idx + len(u)
            spans.append((found_idx, end_idx))
            curr_idx = end_idx
        return spans

    def align(self, text: str) -> AlignedSequence:
        """
        Aligns a single sentence across subword and character channels.
        Returns subword IDs, character IDs, and boundary coordinate arrays start_char_idx and end_char_idx.
        """
        norm_text = normalize_sinhala(text.strip())
        if not norm_text:
            norm_text = " "

        # 1. Subword Tokenization with offset mapping
        subword_encoding = self.subword_tokenizer(
            norm_text,
            truncation=True,
            max_length=self.max_subword_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        subword_ids: List[int] = subword_encoding["input_ids"]
        subword_attn: List[int] = subword_encoding.get("attention_mask", [1] * len(subword_ids))
        subword_tokens: List[str] = self.subword_tokenizer.convert_ids_to_tokens(subword_ids)
        subword_offsets: List[Tuple[int, int]] = subword_encoding["offset_mapping"]

        # 2. Phonological Unit Tokenization
        raw_units = process_text(norm_text)
        unit_spans = self._get_phonological_spans(norm_text, raw_units)

        # Prepend <bos> and append <eos> to phonological units
        char_units = [SinhalaCharTokenizer.BOS_TOKEN] + raw_units + [SinhalaCharTokenizer.EOS_TOKEN]
        char_ids = (
            [self.char_tokenizer.bos_token_id]
            + [self.char_tokenizer.vocab_map.get(u, self.char_tokenizer.unk_token_id) for u in raw_units]
            + [self.char_tokenizer.eos_token_id]
        )

        # Truncate character sequence if needed
        if len(char_ids) > self.max_char_length:
            char_ids = char_ids[: self.max_char_length - 1] + [self.char_tokenizer.eos_token_id]
            char_units = char_units[: self.max_char_length - 1] + [SinhalaCharTokenizer.EOS_TOKEN]
            unit_spans = unit_spans[: len(char_units) - 2]

        char_attn = [1] * len(char_ids)
        bos_idx = 0
        eos_idx = len(char_ids) - 1

        # 3. Align Subword Tokens to Phonological Units
        start_char_idx: List[int] = []
        end_char_idx: List[int] = []

        for i, (tok_start, tok_end) in enumerate(subword_offsets):
            # Special tokens [CLS], [SEP], [PAD]
            if tok_start == 0 and tok_end == 0:
                if i == 0:  # First token is [CLS] / [BOS]
                    start_char_idx.append(bos_idx)
                    end_char_idx.append(bos_idx)
                else:  # Last or padding token is [SEP] / [EOS]
                    start_char_idx.append(eos_idx)
                    end_char_idx.append(eos_idx)
                continue

            # Find matching phonological units by offset overlap
            matching_unit_indices = []
            for u_idx, (u_start, u_end) in enumerate(unit_spans):
                # Check for overlap: max(tok_start, u_start) < min(tok_end, u_end)
                if max(tok_start, u_start) < min(tok_end, u_end):
                    # Offset +1 because index 0 is <bos>
                    matching_unit_indices.append(u_idx + 1)

            if matching_unit_indices:
                start_char_idx.append(matching_unit_indices[0])
                end_char_idx.append(matching_unit_indices[-1])
            else:
                # Nearest fallback to prevent out-of-bounds indexing
                closest_idx = min(
                    range(len(unit_spans)),
                    key=lambda idx: abs(unit_spans[idx][0] - tok_start),
                    default=0,
                )
                adjusted_idx = min(closest_idx + 1, eos_idx - 1)
                start_char_idx.append(adjusted_idx)
                end_char_idx.append(adjusted_idx)

        return AlignedSequence(
            text=norm_text,
            tokens=subword_tokens,
            input_ids=subword_ids,
            attention_mask=subword_attn,
            char_units=char_units,
            char_input_ids=char_ids,
            char_attention_mask=char_attn,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            subword_offsets=subword_offsets,
        )
