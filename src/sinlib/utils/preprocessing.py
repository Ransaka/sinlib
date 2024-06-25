from functools import partial
import multiprocessing
import re
from .chars import VOWEL_DIACRITICS, NUBERS_AND_PUNKTS, ALL_LETTERS
import json
from pathlib import Path
import warnings
from os import path


CURRENT_PATH = path.dirname(path.abspath(__file__))
DEFAULT_VOCAB_MAP_FP = path.join(CURRENT_PATH, 'data')
CHAR_MAPPER_FP = path.join(CURRENT_PATH, 'data', 'char_map.json')


def load_char_mapper(char_mapper_fp):
    if Path(char_mapper_fp).is_file():
        with open(char_mapper_fp, "r") as f:
            char_mapper = json.load(f)
    else:
        warnings.warn(
            "File not found at the specified path. Loaded default char map.",
            UserWarning,
        )
        with open(CHAR_MAPPER_FP, "r") as f:
            char_mapper = json.load(f)
    return char_mapper


def load_default_vocab_map():
    with open(Path(DEFAULT_VOCAB_MAP_FP) / "vocab.json", "r") as f:
        vocab_map = json.load(f)
    return vocab_map


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


def process_text(t):
    tokenized_chars = []

    for i, char in enumerate(t):
        if char in VOWEL_DIACRITICS:
            continue
        if char in NUBERS_AND_PUNKTS:
            tokenized_chars.append(char)
        elif char == " ":
            tokenized_chars.append(" ")
        elif char in ALL_LETTERS:
            if i < len(t) - 1 and t[i + 1] in ALL_LETTERS:
                tokenized_chars.append(char)
            elif i < len(t) - 1 and t[i + 1] in VOWEL_DIACRITICS:
                tokenized_chars.append(char + t[i + 1])
            else:
                tokenized_chars.append(char)
        else:
            tokenized_chars.append(char)

    return tokenized_chars


def process_text_with_token_counts(
    t: str, consider_special_character_as_sinhala: bool, ignore_non_printable: bool
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
        if (char in NUBERS_AND_PUNKTS) and (consider_special_character_as_sinhala):
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
    consider_special_character_as_sinhala: bool = True,
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
            consider_special_character_as_sinhala,
            ignore_non_printable=ignore_non_printable,
        )
        tokenized_text = [tok for tok in tokenized_text if tok != " "]
        return sinhala_token_count / len(tokenized_text)
    elif isinstance(text, list):
        pool = multiprocessing.Pool()
        partial_process_text = partial(
            process_text_with_token_counts,
            consider_special_character_as_sinhala=consider_special_character_as_sinhala,
            ignore_non_printable=ignore_non_printable,
        )
        results = pool.map(partial_process_text, text)
        pool.close()
        pool.join()
        encodings = [tok[0] for tok in results if tok[0] != " "]
        encodings = [[char for char in enc if char != " "] for enc in encodings]
        sinhala_lengths = [tok[1] for tok in results]
        return [(l / len(enc)) for enc, l in zip(encodings, sinhala_lengths)]
