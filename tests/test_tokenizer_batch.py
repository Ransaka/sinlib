"""
Tests for batch encoding and decoding functionality.
"""
import pytest
from sinlib.tokenizer import Tokenizer
from sinlib.encoding import BatchEncoding


@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]


def test_batch_encode_returns_batch_encoding(sample_texts):
    """batch_encode() returns a BatchEncoding whose .input_ids is a list-of-lists."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    result = tokenizer.batch_encode(["මම", "ගෙදර"])
    assert isinstance(result, BatchEncoding)
    assert isinstance(result.input_ids, list)
    assert len(result.input_ids) == 2
    assert isinstance(result.input_ids[0], list)


def test_batch_encode_via_call(sample_texts):
    """Passing a list to __call__ is equivalent to batch_encode."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    result = tokenizer(["මම", "ගෙදර"])
    assert isinstance(result, BatchEncoding)
    assert len(result.input_ids) == 2


def test_batch_encode_matches_individual(sample_texts):
    """Each row of a batch encoding equals the corresponding individual encoding."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    batch = tokenizer.batch_encode(["මම", "ගෙදර"])
    enc1 = tokenizer("මම")
    enc2 = tokenizer("ගෙදර")

    assert batch.input_ids[0] == enc1.input_ids
    assert batch.input_ids[1] == enc2.input_ids


def test_batch_encode_with_padding(sample_texts):
    """With truncate_and_pad, every row is padded to model_max_length."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    batch = tokenizer.batch_encode(["මම", "ගෙදර ගියා"], truncate_and_pad=True)
    assert len(batch.input_ids[0]) == 20
    assert len(batch.input_ids[1]) == 20


def test_batch_encode_with_attention_mask(sample_texts):
    """With return_attention_mask=True, both fields are populated."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    batch = tokenizer.batch_encode(["මම", "ගෙදර"], return_attention_mask=True)
    assert isinstance(batch, BatchEncoding)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    # "මම" has 2 tokens, all real → mask all ones
    assert all(m == 1 for m in batch.attention_mask[0])


def test_batch_decode(sample_texts):
    """batch_decode reproduces the original texts (modulo unknown tokens)."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    enc1 = tokenizer("මම")
    enc2 = tokenizer("ගෙදර")

    decoded = tokenizer.batch_decode([enc1, enc2])
    assert len(decoded) == 2
    assert decoded[0] == "මම"
    assert decoded[1] == "ගෙද<|unk|>"  # 'ර' not in small training vocab


def test_batch_decode_with_attention_mask(sample_texts):
    """batch_decode accepts BatchEncoding objects."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    enc1 = tokenizer("මම", return_attention_mask=True)
    enc2 = tokenizer("ගෙදර", return_attention_mask=True)

    decoded = tokenizer.batch_decode([enc1, enc2])
    assert len(decoded) == 2
    assert decoded[0] == "මම"
    assert decoded[1] == "ගෙද<|unk|>"


def test_empty_batch():
    """Empty list input/output handled gracefully."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(["test"])

    # Empty batch encode
    batch = tokenizer.batch_encode([])
    assert isinstance(batch, BatchEncoding)
    assert batch.input_ids == []

    # Empty batch decode
    decoded = tokenizer.batch_decode([])
    assert decoded == []
