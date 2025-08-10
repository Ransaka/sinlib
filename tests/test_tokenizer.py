import pytest
from pathlib import Path
from sinlib.tokenizer import Tokenizer

@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]

def test_tokenizer_initialization():
    tokenizer = Tokenizer(max_length=100)
    assert tokenizer.max_length == 100
    assert tokenizer.vocab_map is None

def test_train_tokenizer(sample_texts):
    tokenizer = Tokenizer(max_length=50)
    tokenizer.train(sample_texts)
    
    assert tokenizer.vocab_size > 0
    assert "<|unk|>" in tokenizer.vocab_map
    assert "ම" in tokenizer.vocab_map

def test_encode_decode(sample_texts):
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    encoded = tokenizer("මම ගෙදර")
    decoded = tokenizer.decode(encoded)
    
    assert len(encoded) == 5
    assert "මම" in decoded

def test_save_load_tokenizer(tmp_path):
    tokenizer = Tokenizer(max_length=30)
    tokenizer.train(["test", "text"])
    
    save_path = tmp_path / "tokenizer"
    tokenizer.save_tokenizer(save_path)
    
    new_tokenizer = Tokenizer(max_length=30)
    new_tokenizer.load_from_pretrained(save_path, load_default_tokenizer=False)
    
    assert new_tokenizer.vocab_map == tokenizer.vocab_map
    assert new_tokenizer.pad_token_id == tokenizer.pad_token_id

def test_edge_cases():
    tokenizer = Tokenizer(max_length=10)
    tokenizer.train([""])
    
    assert tokenizer("", truncate_and_pad=True) == [tokenizer.pad_token_id] * 10
    assert tokenizer("", truncate_and_pad=False) == []