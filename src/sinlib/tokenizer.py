import json
import warnings
from pathlib import Path
import concurrent.futures
from .utils.preprocessing import process_text, load_default_vocab_map


class Tokenizer:
    def __init__(
        self, max_length: int, unknown_token: str = "<unk>", pad_token: str = "<pad>"
    ):
        self.unknown_token_id = None
        self.token_id_to_token_map = None
        self.vocab_map = None
        self.unknown_token = unknown_token
        self.pad_token = pad_token
        self.tokenized_chars = []
        self.unique_chars = []
        self.special_tokens = [self.unknown_token, self.pad_token]
        self.max_length = max_length
        self.pad_token_id = None

    def __encode(self, text, truncate_and_pad: bool) -> list:
        processed_text = self.__process_text(text)
        text_encodings = [
            self.vocab_map.get(char, self.unknown_token_id) for char in processed_text
        ]
        if truncate_and_pad:
            return self.pad_or_truncate(
                sequence=text_encodings,
                max_length=self.max_length,
                padding_value=self.pad_token_id,
            )
        else:
            return text_encodings

    @staticmethod
    def pad_or_truncate(sequence, max_length, padding_value):
        if len(sequence) > max_length:
            return sequence[:max_length]
        elif len(sequence) < max_length:
            return sequence + [padding_value] * (max_length - len(sequence))
        else:
            return sequence

    def __call__(self, text, truncate_and_pad: bool = True) -> list:
        """
        Encode the given text into a list of tokens.

        Parameters
        ----------
        text : str
            Text to be encoded.
        truncate_and_pad: bool
            Set as True if you need to truncate/pad encodings False otherwise

        Returns
        -------
        encoded_tokens : list of int
            List of tokens representing the encoded text.

        Examples
        --------
        >>> from sinlib import Tokenizer
        >>> corpus = [...]
        >>> tokenizer = Tokenizer()
        >>> tokenizer.train(corpus)
        >>> tokenizer("මම ගෙදර ගියා")
        [2041, 2041, 942, 965, 624, 909, 942, 54, 1960]
        """
        return self.__encode(text, truncate_and_pad=truncate_and_pad)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        """
        Decode a list of token IDs into a string.

        Parameters
        ----------
        ids : list of int
            List of token IDs to be decoded.
        skip_special_tokens: bool
            Whether to consider special tokens when decoding sequences

        Returns
        -------
        decoded_text : str
            The decoded text string.

        Examples
        --------
        >>> from sinlib import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> tokenizer.train([...])
        >>> encoded_tokens = [2041, 2041, 942, 965, 624, 909, 942, 54, 1960]
        >>> tokenizer.decode(encoded_tokens)
        'මම ගෙදර ගියා'
        """
        special_token_ids = [self.vocab_map[tok] for tok in self.special_tokens]
        if skip_special_tokens:
            return "".join(
                [
                    self.token_id_to_token_map.get(token, self.unknown_token)
                    for token in ids
                    if token not in special_token_ids
                ]
            )
        else:
            return "".join(
                [
                    self.token_id_to_token_map.get(token, self.unknown_token)
                    for token in ids
                ]
            )

    def train(self, text_list) -> None:
        """
        Train the tokenizer on a list of text strings.

        Parameters
        ----------
        text_list : list of str
            List of text strings to be used for training the tokenizer.

        Examples
        --------
        >>> from sinlib import Tokenizer
        >>> corpus = [...]
        >>> tokenizer = Tokenizer()
        >>> tokenizer.train(corpus)
        """
        self.__train_character_level_tokenizer(text_list)

    def __len__(self):
        return len(self.vocab_map)

    @property
    def vocab_size(self):
        return len(self)

    @staticmethod
    def __process_text(t):
        return process_text(t)

    def __train_character_level_tokenizer(self, text_list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.__process_text, text_list))
            self.tokenized_chars = [char for sublist in results for char in sublist]
        self.unique_chars = set(self.tokenized_chars)
        self.vocab_map = dict(zip(self.unique_chars, range(len(self.unique_chars))))
        self.vocab_map[self.unknown_token] = len(self.vocab_map)
        self.vocab_map[self.pad_token] = len(self.vocab_map)
        self.unknown_token_id = self.vocab_map[self.unknown_token]
        self.pad_token_id = self.vocab_map[self.pad_token]
        self.token_id_to_token_map = {
            value: key for key, value in self.vocab_map.items()
        }

    def load_from_pretrained(self, file_path: str) -> None:
        """
        Load the vocabulary map from a pre-trained file.

        Parameters
        ----------
        file_path : str
            Path to the file containing the pre-trained vocabulary map.

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If the file is not found at the specified path, a default vocabulary map is loaded and a warning is issued.

        Examples
        --------
        >>> from sinlib import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> tokenizer.load_from_pretrained("pretrained_vocab.json")
        """
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path / "vocab.json", "r") as f:
                self.vocab_map = json.load(f)
            with open(file_path / "config.json", "r") as f:
                configurations = json.load(f)
            self.unknown_token = configurations["unknown_token"]
            self.pad_token = configurations["pad_token"]
            self.unknown_token_id = configurations["unknown_token_id"]
            self.pad_token_id = configurations["pad_token_id"]
            self.max_length = configurations["max_length"]
        else:
            warnings.warn(
                "File not found at the specified path. Loaded default vocab map.",
                UserWarning,
            )
            self.vocab_map = load_default_vocab_map()

        self.token_id_to_token_map = {
            value: key for key, value in self.vocab_map.items()
        }
        self.unknown_token_id = self.vocab_map[self.unknown_token]
        self.pad_token_id = self.vocab_map[self.pad_token]
        return self

    def save_tokenizer(self, save_path: str):
        save_path = Path(save_path)
        configurations = {
            "unknown_token": self.unknown_token,
            "pad_token": self.pad_token,
            "unknown_token_id": self.unknown_token_id,
            "pad_token_id": self.pad_token_id,
            "max_length": self.max_length,
        }

        with open(save_path / "vocab.json", "w", encoding="utf-8") as file:
            json.dump(self.vocab_map, file, ensure_ascii=False, indent=4)

        with open(save_path / "config.json", "w") as file:
            json.dump(configurations, file, indent=4)
