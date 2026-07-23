import pytest
import numpy as np
from sinlib import Tokenizer, SubwordTokenizer
from sinlib.spellcheck import TypoDetector, weighted_levenshtein
from sinlib.utils.preprocessing import normalize_sinhala

# ------------------------------------------------------------------
# 1. Test Tensor Returns
# ------------------------------------------------------------------

def test_tokenizer_return_tensors_numpy():
    tokenizer = Tokenizer(model_max_length=10)
    tokenizer.train(["මම ගෙදර ගියා", "හෙලෝ"])

    res = tokenizer("මම ගෙදර", return_tensors="np")
    assert isinstance(res.input_ids, np.ndarray)
    assert isinstance(res.attention_mask, np.ndarray)
    assert res.input_ids.ndim == 1  # 1D array for single string

    res_batch = tokenizer(["මම ගෙදර", "හෙලෝ"], padding=True, return_tensors="np")
    assert isinstance(res_batch.input_ids, np.ndarray)
    assert isinstance(res_batch.attention_mask, np.ndarray)
    assert res_batch.input_ids.ndim == 2  # 2D array for batch


def test_tokenizer_return_tensors_pytorch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed in this environment.")

    tokenizer = Tokenizer(model_max_length=10)
    tokenizer.train(["මම ගෙදර ගියා", "හෙලෝ"])

    res = tokenizer("මම ගෙදර", return_tensors="pt")
    assert torch.is_tensor(res.input_ids)
    assert torch.is_tensor(res.attention_mask)

    res_batch = tokenizer(["මම ගෙදර", "හෙලෝ"], padding=True, return_tensors="pt")
    assert torch.is_tensor(res_batch.input_ids)
    assert torch.is_tensor(res_batch.attention_mask)
    assert res_batch.input_ids.shape[0] == 2


# ------------------------------------------------------------------
# 2. Test Subword Tokenizer
# ------------------------------------------------------------------

def test_subword_tokenizer(tmp_path):
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError:
        pytest.skip("tokenizers package not installed.")

    corpus = ["මම ගෙදර ගියා", "ඔහු පාසලට ගියා", "සිංහල භාෂාව"]
    sub_tok = SubwordTokenizer.train_from_corpus(corpus, vocab_size=100)

    # Encode / Decode
    encoded = sub_tok.encode("මම ගෙදර")
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)

    decoded = sub_tok.decode(encoded)
    assert "මම" in decoded
    assert "ගෙදර" in decoded

    # Save & Load
    save_path = tmp_path / "subword_vocab.json"
    sub_tok.save(str(save_path))
    assert save_path.exists()

    loaded_tok = SubwordTokenizer.from_file(str(save_path))
    assert loaded_tok.encode("මම ගෙදර") == encoded


# ------------------------------------------------------------------
# 3. Test TypoDetector & Weighted Edit Distance
# ------------------------------------------------------------------

def test_weighted_levenshtein():
    # Identical
    assert weighted_levenshtein("සිංහල", "සිංහල") == 0.0

    # Dental vs Retroflex N (Confusion pair)
    dist_conf = weighted_levenshtein("න", "ණ")
    # Non-confusing, non-adjacent substitution
    dist_reg = weighted_levenshtein("න", "ප")

    assert dist_conf == 0.3
    assert dist_reg == 1.0

    # Keyboard adjacency (e.g. ම and හ in Wijesekera map to u and y)
    dist_key = weighted_levenshtein("ම", "හ")
    assert dist_key == 0.5


def test_spellchecker_reranking():
    # Initialize spell checker (downloads default artifacts)
    detector = TypoDetector.from_pretrained()

    # Suggestion should order phonologically closer candidate higher
    # For example, "අඩිරාජ" is phonologically/morphologically closer to "අධිරාජ"
    suggestions = detector.suggest_correction("අඩිරාජ")
    assert "අධිරාජ" in suggestions

    # Test context-aware correction
    # "අපකරියට ගිය" -> "අපකීර්තියට ගිය" (fits better in the context than other candidate matches)
    corrected = detector("අපකරියට ගිය")
    assert "අපකීර්තියට" in corrected


# ------------------------------------------------------------------
# 4. Test Preprocessing Utilities
# ------------------------------------------------------------------

def test_normalize_sinhala():
    # Remove duplicate diacritics and normalize ZWJ
    text = "ක්‍ර\u200d\u200dය"  # multiple ZWJ
    normalized = normalize_sinhala(text)
    assert "\u200d\u200d" not in normalized

    # Canonicalize composite diacritics (e.g. ෙ + ා -> ො)
    assert normalize_sinhala("කො") == "කො"
    assert normalize_sinhala("කේ") == "කේ"



