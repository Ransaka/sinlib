import concurrent.futures
from .utils.preprocessing import process_text

class Tokenizer:
    def __init__(self):
        self.unknown_token = "<unk>"
        self.tokenized_chars = []
        self.unique_chars = []
    
    def __encode(self, text):
        processed_text = self.__process_text(text)
        encoded_text = [self.vocab_map.get(char, self.unknown_token_id) for char in processed_text]
        return encoded_text
    
    def __call__(self, text):
        return self.__encode(text)
    
    def decode(self, ids):
        return "".join([self.token_id_to_token_map.get(token,self.unknown_token) for token in ids])

    def train(self, text_list):
        self.__train_chracter_level_tokenizer(text_list)
    
    def __len__(self):
        return len(self.vocab_map)

    @staticmethod
    def __process_text(t):
        return process_text(t)

    def __train_chracter_level_tokenizer(self, text_list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.__process_text, text_list))
            self.tokenized_chars = [char for sublist in results for char in sublist]
        self.unique_chars = set(self.tokenized_chars)
        self.vocab_map = dict(zip(self.unique_chars,range(len(self.unique_chars))))
        self.vocab_map[self.unknown_token] = len(self.vocab_map)
        self.unknown_token_id = self.vocab_map[self.unknown_token]
        self.token_id_to_token_map = {value:key for key,value in self.vocab_map.items()}