"""
Tests for BOS (beginning-of-sequence) token handling.
"""
import warnings
import pytest
from sinlib.tokenizer import Tokenizer
from sinlib.encoding import BatchEncoding


@pytest.fixture
def sample_texts():
    return ["මම ගෙදර ගියා", "හෙලෝ වර්ල්ඩ්", "සිංහල අකුරු"]


def test_bos_token_initialization():
    """BOS token is properly set during initialization."""
    tokenizer = Tokenizer(model_max_length=100, bos_token="<|start|>")
    assert tokenizer.bos_token == "<|start|>"
    assert "<|start|>" in tokenizer.special_tokens


def test_bos_token_in_vocabulary(sample_texts):
    """BOS token is assigned ID 3 after training (pad=0, unk=1, eos=2, bos=3)."""
    tokenizer = Tokenizer(model_max_length=50)
    tokenizer.train(sample_texts)

    assert tokenizer.bos_token_id == 3
    assert tokenizer.vocab_map[tokenizer.bos_token] == 3


def test_encode_with_bos_token(sample_texts):
    """add_bos_token=True prepends BOS and extends the sequence by 1."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    without_bos = tokenizer("මම ගෙදර")
    with_bos = tokenizer("මම ගෙදර", add_bos_token=True)

    assert len(with_bos.input_ids) == len(without_bos.input_ids) + 1
    assert with_bos.input_ids[0] == tokenizer.bos_token_id
    assert with_bos.input_ids[1:] == without_bos.input_ids


def test_batch_encode_with_bos_token(sample_texts):
    """BOS token is prepended to each sequence in a batch."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    texts = ["මම", "ගෙදර"]
    without_bos = tokenizer.batch_encode(texts)
    with_bos = tokenizer.batch_encode(texts, add_bos_token=True)

    for i in range(len(texts)):
        assert len(with_bos.input_ids[i]) == len(without_bos.input_ids[i]) + 1
        assert with_bos.input_ids[i][0] == tokenizer.bos_token_id
        assert with_bos.input_ids[i][1:] == without_bos.input_ids[i]


def test_decode_with_bos_token(sample_texts):
    """skip_special_tokens=True strips BOS from decoded output."""
    tokenizer = Tokenizer(model_max_length=20)
    tokenizer.train(sample_texts)

    text = "මම ගෙදර"
    encoded_with_bos = tokenizer(text, add_bos_token=True)

    # With special tokens preserved
    decoded_with = tokenizer.decode(encoded_with_bos, skip_special_tokens=False)
    assert tokenizer.bos_token in decoded_with

    # With special tokens skipped
    decoded_without = tokenizer.decode(encoded_with_bos, skip_special_tokens=True)
    assert tokenizer.bos_token not in decoded_without
    # 'ර' is not in the small training corpus → unknown token (also skipped)
    assert decoded_without == text.replace("ර", "")


def test_save_load_with_bos_token(tmp_path):
    """Custom BOS token is preserved across save/load round-trip."""
    tokenizer = Tokenizer(model_max_length=30, bos_token="<|custom_bos|>")
    tokenizer.train(["test", "text"])

    save_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(save_path))

    loaded = Tokenizer.from_pretrained(str(save_path))

    assert loaded.bos_token == "<|custom_bos|>"
    assert loaded.bos_token_id == tokenizer.bos_token_id

    encoded = loaded("test", add_bos_token=True)
    assert encoded.input_ids[0] == loaded.bos_token_id


def test_save_load_legacy_bos_token(tmp_path):
    """Legacy save_tokenizer / load_from_pretrained preserves BOS token."""
    tokenizer = Tokenizer(model_max_length=30, bos_token="<|custom_bos|>")
    tokenizer.train(["test", "text"])

    save_path = tmp_path / "tokenizer"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tokenizer.save_tokenizer(str(save_path))

    new_tok = Tokenizer(model_max_length=30)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        new_tok.load_from_pretrained(str(save_path), load_default_tokenizer=False)

    assert new_tok.bos_token == "<|custom_bos|>"
    assert new_tok.bos_token_id == tokenizer.bos_token_id

    encoded = new_tok("test", add_bos_token=True)
    assert encoded.input_ids[0] == new_tok.bos_token_id
