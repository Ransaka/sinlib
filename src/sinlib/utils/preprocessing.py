import multiprocessing
import re
from .chars import VOWEL_DIACRITICS, NUBERS_AND_PUNKTS, ALL_LETTERS
import numpy as np
import os

# file_path = os.path.join(os.path.dirname(__file__), '../data', 'sinhala_chars_with_special_chars.txt')

# with open(file_path,'r') as f:
#     SINHALA_CHARS_WITH_SPECIAL_CHARS = f.read().split("\n")


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


def process_text_with_token_counts(t, consider_special_character_as_sinhala):
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


def get_sinhala_character_ratio(text, consider_special_character_as_sinhala:bool=True, ignore_non_printable:bool=True):
    """Retuning sinhala character ratio for given text string for given settings."""
    if ignore_non_printable:
        text = remove_non_printable(text)
    if isinstance(text, str):
        tokenized_text, sinhala_token_count = process_text_with_token_counts(text,consider_special_character_as_sinhala)
        tokenized_text = [tok for tok in tokenized_text if tok != " "]
        return sinhala_token_count / len(tokenized_text)
    elif isinstance(text, list):
        pool = multiprocessing.Pool()
        results = pool.map(process_text_with_token_counts, text)
        pool.close()
        pool.join()
        encodings = [tok[0] for tok in results if tok[0] != " "]
        encodings = [[char for char in enc if char != " "] for enc in encodings]
        sinhala_lengths = [tok[1] for tok in results]
        return [(l / len(enc)) for enc, l in zip(encodings, sinhala_lengths)]
