import concurrent.futures
from utils.chars import VOWEL_DIACRITICS, NUMERALS, NUBERS_AND_PUNKTS, ALL_LETTERS
from utils.preprocessing import remove_non_printable

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

    def __process_text(self, t):
        # t = remove_non_printable(t)
        tokenized_chars = []

        for i, char in enumerate(t):
            if char in VOWEL_DIACRITICS:
                continue
            if char in NUBERS_AND_PUNKTS:
                tokenized_chars.append(char)
            elif char == ' ':
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

    def __train_chracter_level_tokenizer(self, text_list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.__process_text, text_list))
            self.tokenized_chars = [char for sublist in results for char in sublist]
        self.unique_chars = set(self.tokenized_chars)
        self.vocab_map = dict(zip(self.unique_chars,range(len(self.unique_chars))))
        self.vocab_map[self.unknown_token] = len(self.vocab_map)
        self.unknown_token_id = self.vocab_map[self.unknown_token]
        self.token_id_to_token_map = {value:key for key,value in self.vocab_map.items()}