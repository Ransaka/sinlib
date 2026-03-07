"""
Tests for special tokens and attention mask behavior.
"""
import pytest
from sinlib.tokenizer import Tokenizer
from sinlib.encoding import BatchEncoding


@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]


def test_special_tokens_at_beginning(sample_texts):
    """Special tokens occupy fixed IDs 0-3 in the vocabulary."""
    tokenizer = Tokenizer(model_max_length=50)
    tokenizer.train(sample_texts)

    assert tokenizer.pad_token_id == 0
    # Both new and legacy ID attributes
    assert tokenizer.unk_token_id == 1
    assert tokenizer.unknown_token_id == 1  # legacy alias
    assert tokenizer.eos_token_id == 2
    assert tokenizer.end_of_text_token_id == 2  # legacy alias
    assert tokenizer.bos_token_id == 3

    # Verify via vocab_map lookup using both new and legacy token attributes
    assert tokenizer.vocab_map[tokenizer.pad_token] == 0
    assert tokenizer.vocab_map[tokenizer.unk_token] == 1
    assert tokenizer.vocab_map[tokenizer.unknown_token] == 1  # legacy alias
    assert tokenizer.vocab_map[tokenizer.eos_token] == 2
    assert tokenizer.vocab_map[tokenizer.end_of_text_token] == 2  # legacy alias


def test_attention_mask_no_padding(sample_texts):
    """Without padding, attention mask is all 1s."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    result = tokenizer("මම", return_attention_mask=True)
    assert isinstance(result, BatchEncoding)
    assert "input_ids" in result
    assert "attention_mask" in result
    # No padding → all tokens are real
    assert all(m == 1 for m in result.attention_mask)


def test_attention_mask_with_padding(sample_texts):
    """Padded positions have mask=0; real token positions have mask=1."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    result = tokenizer("මම", truncate_and_pad=True, return_attention_mask=True)
    input_ids = result.input_ids
    attention_mask = result.attention_mask

    # "මම" tokenizes to exactly 2 tokens
    assert sum(attention_mask) == 2

    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.pad_token_id:
            assert attention_mask[i] == 0
        else:
            assert attention_mask[i] == 1


def test_decode_with_batch_encoding(sample_texts):
    """decode() accepts BatchEncoding directly."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    text = "මම ගෙදර"
    result = tokenizer(text, return_attention_mask=True)
    decoded = tokenizer.decode(result)

    assert decoded == text.replace("ර", "<|unk|>")


def test_decode_with_dict_legacy(sample_texts):
    """decode() accepts a plain dict with 'input_ids' key (legacy)."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    text = "මම"
    enc = tokenizer(text)
    legacy_dict = enc.to_dict()

    decoded = tokenizer.decode(legacy_dict)
    assert decoded == text


def test_encode_plus(sample_texts):
    """encode_plus returns BatchEncoding with both fields."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    enc = tokenizer.encode_plus("මම", padding=True, max_length=10)
    assert isinstance(enc, BatchEncoding)
    assert len(enc.input_ids) == 10
    assert sum(enc.attention_mask) == 2  # 2 real tokens
