"""
Tests for the Tokenizer class — basic initialization, training, encode/decode, save/load.
"""
import pytest
import warnings
from pathlib import Path
from sinlib.tokenizer import Tokenizer
from sinlib.encoding import BatchEncoding


@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]


def test_tokenizer_initialization():
    tokenizer = Tokenizer(model_max_length=100)
    assert tokenizer.model_max_length == 100
    assert tokenizer.max_length == 100  # legacy alias
    assert tokenizer.vocab_map is None


def test_tokenizer_initialization_legacy_kwarg():
    """max_length= still accepted as legacy kwarg."""
    tokenizer = Tokenizer(max_length=100)
    assert tokenizer.model_max_length == 100


def test_train_tokenizer(sample_texts):
    tokenizer = Tokenizer(model_max_length=50)
    tokenizer.train(sample_texts)

    assert tokenizer.vocab_size > 0
    assert "<|unk|>" in tokenizer.vocab_map
    assert "ම" in tokenizer.vocab_map


def test_encode_returns_batch_encoding(sample_texts):
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    result = tokenizer("මම ගෙදර")
    assert isinstance(result, BatchEncoding)
    assert "input_ids" in result
    # attention_mask is always included by default
    assert "attention_mask" in result


def test_encode_decode(sample_texts):
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    encoded = tokenizer("මම ගෙදර")
    decoded = tokenizer.decode(encoded)

    assert len(encoded.input_ids) == 5
    assert "මම" in decoded


def test_encode_method(sample_texts):
    """Tokenizer.encode() returns a plain List[int]."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    ids = tokenizer.encode("මම")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_tokenize_method(sample_texts):
    """Tokenizer.tokenize() returns List[str] of character tokens."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    tokens = tokenizer.tokenize("ආයුබෝවන්")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    # Diacritic should be combined with base consonant
    assert "යු" in tokens
    assert "බෝ" in tokens


def test_convert_tokens_to_ids(sample_texts):
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    # Single token
    id_ = tokenizer.convert_tokens_to_ids("ම")
    assert isinstance(id_, int)

    # List of tokens
    ids = tokenizer.convert_tokens_to_ids(["ම", "ම"])
    assert isinstance(ids, list)
    assert len(ids) == 2


def test_convert_ids_to_tokens(sample_texts):
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    # Single id
    tok = tokenizer.convert_ids_to_tokens(4)
    assert isinstance(tok, str)

    # List
    toks = tokenizer.convert_ids_to_tokens([4, 4])
    assert isinstance(toks, list)


def test_get_vocab(sample_texts):
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    vocab = tokenizer.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) == tokenizer.vocab_size


def test_all_special_tokens(sample_texts):
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    specials = tokenizer.all_special_tokens
    assert "<|pad|>" in specials
    assert "<|unk|>" in specials
    assert "<|end_of_text|>" in specials
    assert "<|bos|>" in specials


def test_save_load_tokenizer_save_pretrained(tmp_path, sample_texts):
    tokenizer = Tokenizer(model_max_length=30)
    tokenizer.train(sample_texts)

    save_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(save_path))

    loaded = Tokenizer.from_pretrained(str(save_path))
    assert loaded.vocab_map == tokenizer.vocab_map
    assert loaded.pad_token_id == tokenizer.pad_token_id


def test_save_load_tokenizer_legacy(tmp_path, sample_texts):
    """Legacy save_tokenizer + load_from_pretrained still works (with deprecation warnings)."""
    tokenizer = Tokenizer(model_max_length=30)
    tokenizer.train(sample_texts)

    save_path = tmp_path / "tokenizer"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tokenizer.save_tokenizer(str(save_path))

    new_tokenizer = Tokenizer(model_max_length=30)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        new_tokenizer.load_from_pretrained(str(save_path), load_default_tokenizer=False)

    assert new_tokenizer.vocab_map == tokenizer.vocab_map
    assert new_tokenizer.pad_token_id == tokenizer.pad_token_id


def test_edge_case_empty_string():
    tokenizer = Tokenizer(model_max_length=10)
    tokenizer.train([""])

    # With padding: should return pad tokens only
    result_padded = tokenizer("", truncate_and_pad=True)
    assert isinstance(result_padded, BatchEncoding)
    assert result_padded.input_ids == [tokenizer.pad_token_id] * 10

    # Without padding: empty input_ids
    result_unpadded = tokenizer("", truncate_and_pad=False)
    assert isinstance(result_unpadded, BatchEncoding)
    assert result_unpadded.input_ids == []


def test_repr():
    tokenizer = Tokenizer(model_max_length=32)
    r = repr(tokenizer)
    assert "Tokenizer" in r
    assert "trained=False" in r
