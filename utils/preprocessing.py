import re


with open("../data/sinhala_chars_with_special_chars.txt",'r') as f:
    SINHALA_CHARS_WITH_SPECIAL_CHARS = f.read().split("\n")

def remove_non_printable(input_string):
    printable_pattern = re.compile(r'[^\u0020-\u007E\u0D80-\u0DFF]+', flags=re.UNICODE)
    return printable_pattern.sub('', input_string)

def remove_english_characters(text):
    """
    Remove English characters from the given text using a regular expression.

    Parameters:
    - text (str): The input text containing English and Sinhala characters.

    Returns:
    - str: Text with English characters removed.
    """
    english_pattern = re.compile("[a-zA-Z]")
    text = english_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retain_sinhala_characters(text):
    """
    Remove non-Sinhala characters from the given text using a conditional expression.

    Parameters:
    - text (str): The input text containing mixed languages.
    Returns:
    - str: Text with Sinhala language.
    """
    input_string = "".join([char for char in text if char in SINHALA_CHARS_WITH_SPECIAL_CHARS])
    cleaned_string = re.sub(r'\s+', ' ', input_string).strip()
    return cleaned_string
