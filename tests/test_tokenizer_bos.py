import pytest
from sinlib.tokenizer import Tokenizer

@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]

def test_bos_token_initialization():
    """Test that BOS token is properly initialized."""
    tokenizer = Tokenizer(max_length=100, bos_token="<|start|>")
    assert tokenizer.bos_token == "<|start|>"
    assert "<|start|>" in tokenizer.special_tokens

def test_bos_token_in_vocabulary(sample_texts):
    """Test that BOS token is included in the vocabulary after training."""
    tokenizer = Tokenizer(max_length=50)
    tokenizer.train(sample_texts)
    
    # Check that BOS token has the expected ID (should be 3 after pad=0, unk=1, eos=2)
    assert tokenizer.bos_token_id == 3
    assert tokenizer.vocab_map[tokenizer.bos_token] == 3

def test_encode_with_bos_token(sample_texts):
    """Test encoding with BOS token."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    # Encode without BOS token
    encoded_without_bos = tokenizer("මම ගෙදර")
    
    # Encode with BOS token
    encoded_with_bos = tokenizer("මම ගෙදර", add_bos_token=True)
    
    # The sequence with BOS should be one token longer
    assert len(encoded_with_bos) == len(encoded_without_bos) + 1
    
    # The first token should be the BOS token
    assert encoded_with_bos[0] == tokenizer.bos_token_id
    
    # The rest of the sequence should match the original encoding
    assert encoded_with_bos[1:] == encoded_without_bos

def test_batch_encode_with_bos_token(sample_texts):
    """Test batch encoding with BOS token."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    texts = ["මම", "ගෙදර"]
    
    # Batch encode without BOS token
    batch_encoded_without_bos = tokenizer.batch_encode(texts)
    
    # Batch encode with BOS token
    batch_encoded_with_bos = tokenizer.batch_encode(texts, add_bos_token=True)
    
    # Check each sequence in the batch
    for i in range(len(texts)):
        # The sequence with BOS should be one token longer
        assert len(batch_encoded_with_bos[i]) == len(batch_encoded_without_bos[i]) + 1
        
        # The first token should be the BOS token
        assert batch_encoded_with_bos[i][0] == tokenizer.bos_token_id
        
        # The rest of the sequence should match the original encoding
        assert batch_encoded_with_bos[i][1:] == batch_encoded_without_bos[i]

def test_decode_with_bos_token(sample_texts):
    """Test decoding with BOS token."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    text = "මම ගෙදර"
    
    # Encode with BOS token
    encoded_with_bos = tokenizer(text, add_bos_token=True)
    
    # Decode with special tokens
    decoded_with_special = tokenizer.decode(encoded_with_bos, skip_special_tokens=False)
    
    # Decode without special tokens
    decoded_without_special = tokenizer.decode(encoded_with_bos, skip_special_tokens=True)
    
    # With special tokens, the BOS token should be included
    assert tokenizer.bos_token in decoded_with_special
    
    # Without special tokens, the BOS token should be skipped
    assert tokenizer.bos_token not in decoded_without_special
    assert decoded_without_special == text.replace("ර", "") # because we use <|unk|> for unknown tokens and set skip_special_tokens=True

def test_save_load_with_bos_token(tmp_path):
    """Test saving and loading tokenizer with BOS token."""
    tokenizer = Tokenizer(max_length=30, bos_token="<|custom_bos|>")
    tokenizer.train(["test", "text"])
    
    save_path = tmp_path / "tokenizer"
    tokenizer.save_tokenizer(save_path)
    
    new_tokenizer = Tokenizer(max_length=30)
    new_tokenizer.load_from_pretrained(save_path, load_default_tokenizer=False)
    
    # Check that BOS token and its ID are preserved
    assert new_tokenizer.bos_token == "<|custom_bos|>"
    assert new_tokenizer.bos_token_id == tokenizer.bos_token_id
    
    # Test encoding with the loaded tokenizer
    encoded = new_tokenizer("test", add_bos_token=True)
    assert encoded[0] == new_tokenizer.bos_token_id