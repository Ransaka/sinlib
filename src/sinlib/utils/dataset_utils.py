from sinlib.tokenizer import Tokenizer

MAX_LENGTH = 32
DUMMY_FILE_NAME = "vocab"


def load_tokenizer():
    tokenizer = Tokenizer(max_length=MAX_LENGTH)
    tokenizer.load_from_pretrained(DUMMY_FILE_NAME)
    return tokenizer


