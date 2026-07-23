from sinlib.tokenizer import Tokenizer

MAX_LENGTH = 32
DUMMY_FILE_NAME = "vocab"


def load_tokenizer():
    return Tokenizer.from_pretrained("Ransaka/sinlib", model_max_length=MAX_LENGTH)


