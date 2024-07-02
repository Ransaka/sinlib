from sinlib.utils.model_utils import load_transliterator_model, inference
from sinlib.utils.dataset_utils import load_tokenizer


class Transliterator:
    def __init__(self):
        self.model = load_transliterator_model()
        self.tokenizer = load_tokenizer()

    def transliterate(self, text):
        word_list = text.split()
        transliterated_text = [inference(self.model, self.tokenizer, word) for word in word_list]
        return " ".join(transliterated_text).strip()
