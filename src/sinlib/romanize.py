from .utils.preprocessing import load_char_mapper
from .tokenizer import Tokenizer
from .utils.preprocessing import DEFAULT_VOCAB_MAP_FP, CHAR_MAPPER_FP
from .utils.chars import ALL_SINHALA_CHARACTERS, NUBERS_AND_PUNKTS
from .utils.preprocessing import remove_non_printable
import numpy as np


class Romanizer:
    def __init__(self, char_mapper_fp: str, tokenizer_path: str):
        if char_mapper_fp is None:
            char_mapper_fp = CHAR_MAPPER_FP
        if tokenizer_path is None:
            tokenizer_path = DEFAULT_VOCAB_MAP_FP
        self.char_mapper = load_char_mapper(char_mapper_fp)
        self.tokenizer = Tokenizer(max_length=None)
        self.tokenizer.load_from_pretrained(tokenizer_path)

    def __call__(self, text):
        return self.__romanize(text)

    def __romanize(self, text: str):
        text = remove_non_printable(text)
        chars = np.array(list(text))
        sinhala_mask = [
            True
            if ch in ALL_SINHALA_CHARACTERS + list(NUBERS_AND_PUNKTS) + [" "]
            else False
            for ch in chars
        ]
        sinhala_text = "".join(chars[sinhala_mask]).strip()
        encodings = self.tokenizer(sinhala_text, truncate_and_pad=False)
        decoded_sinhala_chars = [
            self.tokenizer.token_id_to_token_map[c] for c in encodings
        ]
        romanized_sinhala = [
            self.char_mapper.get(ch, ch if ch in NUBERS_AND_PUNKTS.union(" ") else '')
            for ch in decoded_sinhala_chars
        ]
        romanized_sinhala = "".join(romanized_sinhala)
        word_2_word_mapping = dict(zip(sinhala_text.split(), romanized_sinhala.split()))
        romanized_text = [word_2_word_mapping.get(word, word) for word in text.split()]
        return " ".join(romanized_text)
