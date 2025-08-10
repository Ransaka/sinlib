from functools import partial
import multiprocessing
import re
from .chars import VOWEL_DIACRITICS, NUMBERS_AND_PUNCTUATION, ALL_LETTERS
import json
from pathlib import Path
from enum import Enum

class Filenames(Enum):
    """Enumeration for consistent filename references."""
    VOCAB = "vocab.json"
    CONFIG = "config.json"
    CHAR_MAPPER = "char_map.json"
    NGRAM_PROBS = "ngram_probs.npy"
    DICTIONARY = "dictionary.npy"

def download_hub_file(file_name:str):
    from huggingface_hub.file_download import hf_hub_download
    return hf_hub_download(
        repo_id="Ransaka/sinlib",
        filename=file_name,
        repo_type="model",
    )

def load_char_mapper():
    char_mapper_fp = download_hub_file(Filenames.CHAR_MAPPER.value)
    if Path(char_mapper_fp).is_file():
        with open(char_mapper_fp, "r") as f:
            char_mapper = json.load(f)
    else:
        raise ValueError(
            "File not found at the specified path. Loaded default char map.",
            UserWarning,
        )
    return char_mapper


def load_default_vocab_map():
    file_path = download_hub_file(Filenames.VOCAB.value)
    with open(file_path, "r") as f:
        vocab_map = json.load(f)
    return vocab_map

def load_default_config():
    file_path = download_hub_file(Filenames.CONFIG.value)
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


def remove_non_printable(input_string):
    printable_pattern = re.compile(r"[^\u0020-\u007E\u0D80-\u0DFF]+", flags=re.UNICODE)
    return printable_pattern.sub("", input_string)


def remove_english_characters(text):
    """
    Remove English characters from the given text using a regular expression.

    Parameters:
    - text (str): The input text containing English and Sinhala characters.

    Returns:
    - str: Text with English characters removed.
    """
    english_pattern = re.compile("[a-zA-Z]")
    text = english_pattern.sub(r"", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# def retain_sinhala_characters(text):
#     """
#     Remove non-Sinhala characters from the given text using a conditional expression.

#     Parameters:
#     - text (str): The input text containing mixed languages.
#     Returns:
#     - str: Text with Sinhala language.
#     """
#     input_string = "".join([char for char in text if char in SINHALA_CHARS_WITH_SPECIAL_CHARS])
#     cleaned_string = re.sub(r'\s+', ' ', input_string).strip()
#     return cleaned_string


# def process_text(t):
#     tokenized_chars = []

#     for i, char in enumerate(t):
#         if char in VOWEL_DIACRITICS:
#             continue
#         if char in NUMBERS_AND_PUNCTUATION:
#             tokenized_chars.append(char)
#         elif char == " ":
#             if i < len(t) - 1 and t[i + 1] in VOWEL_DIACRITICS+list(ALL_LETTERS):
#                 tokenized_chars[-1] = tokenized_chars[-1] + " " # space between words
#             else:
#                 tokenized_chars.append(char)
#         elif char in ALL_LETTERS:
#             if i < len(t) - 1 and t[i + 1] in ALL_LETTERS:
#                 tokenized_chars.append(char)
#             elif i < len(t) - 1 and t[i + 1] in VOWEL_DIACRITICS:
#                 tokenized_chars.append(char + t[i + 1])
#             else:
#                 tokenized_chars.append(char)
#         else:
#             tokenized_chars.append(char)

#     return tokenized_chars

def process_text(text: str) -> list[str]:
    """
    Tokenizes text by combining consonants with diacritics and joining a 
    space to the preceding token if it's followed by a letter.

    Args:
        text: The input string to tokenize.

    Returns:
        A list of string tokens. Each token is either a single character,
        a combination of a letter and a diacritic, or a space.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    tokens = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        if char == ' ':
            # Check if the list is not empty AND if the space is followed by a letter/diacritic
            if tokens and i + 1 < n and text[i + 1] in VOWEL_DIACRITICS + list(ALL_LETTERS):
                # Safely append the space to the last token
                tokens[-1] += ' '
            else:
                # Otherwise, treat the space as a separate token
                tokens.append(char)
            i += 1
        
        # Logic for combining a letter with a following diacritic
        elif char in ALL_LETTERS:
            if i + 1 < n and text[i + 1] in VOWEL_DIACRITICS:
                tokens.append(char + text[i + 1])
                i += 2  # Consumed two characters
            else:
                tokens.append(char)
                i += 1  # Consumed one character
        else:
            tokens.append(char)
            i += 1

    return tokens


def process_text_with_token_counts(
    t: str, ignore_punctuation_and_numbers: bool, ignore_non_printable: bool
):
    """
    Process the given text, tokenizing it and counting the tokens.

    Parameters
    ----------
    t : str
        The text to be processed.
    consider_special_character_as_sinhala : bool
        If True, special characters will be considered as Sinhala characters.
    ignore_non_printable : bool
        If True, non-printable characters will be removed from the text.

    Returns
    -------
    tokenized_chars : list of str
        List of tokenized characters from the text.
    token_counts : int
        Total count of tokens in the text.

    Examples
    --------
    >>> from sinlib.utils.preprocessing import process_text_with_token_counts
    >>> text = "මම ගෙදර ගියා."
    >>> tokenized_chars, token_counts = process_text_with_token_counts(text, True, True)
    >>> print(tokenized_chars)
    ['ම', 'ම', ' ', 'ගෙ', 'ද', 'ර', ' ', 'ගි', 'යා', '.']
    >>> print(token_counts)
    10
    """
    if ignore_non_printable:
        t = remove_non_printable(t)

    tokenized_chars = []
    token_counts = 0

    for i, char in enumerate(t):
        if char in VOWEL_DIACRITICS:
            continue
        if (char in NUMBERS_AND_PUNCTUATION) and (ignore_punctuation_and_numbers):
            tokenized_chars.append(char)
            token_counts += 1
        elif char == " ":
            tokenized_chars.append(" ")
        elif char in ALL_LETTERS:
            token_counts += 1
            if i < len(t) - 1 and t[i + 1] in ALL_LETTERS:
                tokenized_chars.append(char)
            elif i < len(t) - 1 and t[i + 1] in VOWEL_DIACRITICS:
                tokenized_chars.append(char + t[i + 1])
            else:
                tokenized_chars.append(char)
        else:
            tokenized_chars.append(char)

    return tokenized_chars, token_counts


def get_sinhala_character_ratio(
    text,
    ignore_punctuation_and_numbers: bool = True,
    ignore_non_printable: bool = True,
):
    """
    Calculate the ratio of Sinhala characters in the given text.

    Parameters
    ----------
    text : str or list of str
        The text or list of text strings to be processed.
    consider_special_character_as_sinhala : bool, default=True
        If True, numbers and special characters will be considered as Sinhala characters.
    ignore_non_printable : bool, default=True
        If True, non-printable characters will be removed before processing.

    Returns
    -------
    ratio : float or list of float
        The ratio of Sinhala characters in the text. If the input is a list, returns a list of ratios for each text string.

    Examples
    --------
    >>> from sinlib.utils.preprocessing import get_sinhala_character_ratio
    >>> text = "මම ගෙදර ගියා."
    >>> ratio = get_sinhala_character_ratio(text, True, True)
    >>> print(ratio)
    1.0

    >>> texts = ["මම ගෙදර ගියා.", "This is an example."]
    >>> ratio = get_sinhala_character_ratio(texts, False, True)
    >>> print(ratios)
    [0.875, 0.0]
    """
    if isinstance(text, str):
        tokenized_text, sinhala_token_count = process_text_with_token_counts(
            text,
            ignore_punctuation_and_numbers,
            ignore_non_printable=ignore_non_printable,
        )
        tokenized_text = [tok for tok in tokenized_text if tok != " "]
        return sinhala_token_count / len(tokenized_text)
    elif isinstance(text, list):
        pool = multiprocessing.Pool()
        partial_process_text = partial(
            process_text_with_token_counts,
            ignore_punctuation_and_numbers=ignore_punctuation_and_numbers,
            ignore_non_printable=ignore_non_printable,
        )
        results = pool.map(partial_process_text, text)
        pool.close()
        pool.join()
        encodings = [tok[0] for tok in results if tok[0] != " "]
        encodings = [[char for char in enc if char != " "] for enc in encodings]
        sinhala_lengths = [tok[1] for tok in results]
        return [(l / len(enc)) for enc, l in zip(encodings, sinhala_lengths)]
