import pytest
from sinlib.tokenizer import Tokenizer

@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]

def test_batch_encode(sample_texts):
    """Test batch encoding functionality."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    # Test basic batch encoding
    batch_encoded = tokenizer.batch_encode(["මම", "ගෙදර"])
    assert len(batch_encoded) == 2
    assert isinstance(batch_encoded, list)
    assert isinstance(batch_encoded[0], list)
    
    # Verify the encoded tokens match individual encoding
    individual_encoded1 = tokenizer("මම")
    individual_encoded2 = tokenizer("ගෙදර")
    assert batch_encoded[0] == individual_encoded1
    assert batch_encoded[1] == individual_encoded2

def test_batch_encode_with_options(sample_texts):
    """Test batch encoding with various options."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    # Test with truncation and padding
    batch_encoded = tokenizer.batch_encode(["මම", "ගෙදර ගියා"], truncate_and_pad=True)
    assert len(batch_encoded) == 2
    assert len(batch_encoded[0]) == 20  # Should be padded to max_length
    assert len(batch_encoded[1]) == 20  # Should be padded to max_length
    
    # Test with attention mask
    batch_encoded = tokenizer.batch_encode(["මම", "ගෙදර"], return_attention_mask=True)
    assert len(batch_encoded) == 2
    assert "input_ids" in batch_encoded[0]
    assert "attention_mask" in batch_encoded[0]
    assert all(mask == 1 for mask in batch_encoded[0]["attention_mask"])

def test_batch_decode(sample_texts):
    """Test batch decoding functionality."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    # Encode some texts
    encoded1 = tokenizer("මම")
    encoded2 = tokenizer("ගෙදර")
    
    # Test batch decoding
    decoded_texts = tokenizer.batch_decode([encoded1, encoded2])
    assert len(decoded_texts) == 2
    assert decoded_texts[0] == "මම"
    assert decoded_texts[1] == "ගෙද<|unk|>" # cause 'ර' not in training vocab.

def test_batch_decode_with_attention_mask(sample_texts):
    """Test batch decoding with attention mask."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(sample_texts)
    
    # Encode with attention mask
    encoded1 = tokenizer("මම", return_attention_mask=True)
    encoded2 = tokenizer("ගෙදර", return_attention_mask=True)
    
    # Test batch decoding with attention mask
    decoded_texts = tokenizer.batch_decode([encoded1, encoded2])
    assert len(decoded_texts) == 2
    assert decoded_texts[0] == "මම"
    assert decoded_texts[1] == "ගෙද<|unk|>"

def test_empty_batch():
    """Test handling of empty batch."""
    tokenizer = Tokenizer(max_length=20)
    tokenizer.train(["test"])
    
    # Test empty batch encoding
    batch_encoded = tokenizer.batch_encode([])
    assert batch_encoded == []
    
    # Test empty batch decoding
    decoded_texts = tokenizer.batch_decode([])
    assert decoded_texts == []