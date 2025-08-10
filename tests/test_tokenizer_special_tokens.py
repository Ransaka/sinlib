import pytest
from sinlib.tokenizer import Tokenizer

@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]

def test_special_tokens_at_beginning(sample_texts):
    """Test that special tokens are at the beginning of the vocabulary."""
    tokenizer = Tokenizer(max_length=50)
    tokenizer.train(sample_texts)
    
    # Check that special tokens have the expected IDs
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unknown_token_id == 1
    assert tokenizer.end_of_text_token_id == 2
    
    # Verify the tokens in the vocabulary map
    assert tokenizer.vocab_map[tokenizer.pad_token] == 0
    assert tokenizer.vocab_map[tokenizer.unknown_token] == 1
    assert tokenizer.vocab_map[tokenizer.end_of_text_token] == 2

def test_attention_mask(sample_texts):
    """Test that attention mask is correctly generated."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    # Test with no padding (all 1s in attention mask)
    text = "මම"
    result = tokenizer(text, return_attention_mask=True)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert all(mask == 1 for mask in result["attention_mask"])
    
    # Test with padding (should have 1s for tokens and 0s for padding)
    result = tokenizer(text, truncate_and_pad=True, return_attention_mask=True)
    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    
    # Check that we have the right number of 1s (for actual tokens)
    assert sum(attention_mask) == 2  # "මම" is 2 characters
    
    # Check that padding tokens have 0 in attention mask
    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.pad_token_id:
            assert attention_mask[i] == 0
        else:
            assert attention_mask[i] == 1

def test_decode_with_attention_mask(sample_texts):
    """Test decoding when input includes attention mask."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    text = "මම ගෙදර"
    result = tokenizer(text, return_attention_mask=True)
    decoded = tokenizer.decode(result)
    
    assert decoded == text.replace("ර", "<|unk|>")