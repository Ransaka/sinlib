import pytest
from sinlib.spellcheck import TypoDetector, _HAS_TORCH

def test_typo_detector_initialization():
    detector = TypoDetector(lazy_loading=True)
    assert detector._akshara_ngram is None
    detector._ensure_loaded()
    assert detector._akshara_ngram is not None
    assert detector._akshara_vocab is not None
    if _HAS_TORCH:
        assert detector._bigru_detector is not None
        assert detector.has_neural_labeler is True

def test_word_suspicion_checking():
    detector = TypoDetector()
    
    # Correct words should not be flagged as suspicious
    assert not detector.is_word_suspicious("පාසලට")
    assert not detector.is_word_suspicious("ගුරුවරයා")
    assert not detector.is_word_suspicious("සිංහල")
    
    # Typo words should be flagged as suspicious
    assert detector.is_word_suspicious("පසලට")
    assert detector.is_word_suspicious("උගන්වය්")
    assert detector.is_word_suspicious("සින්හල")

def test_spellcheck_correction():
    detector = TypoDetector()
    
    # Check that typo correction resolves correctly
    corrected = detector("ගුරුවරයා අපට උගන්වය්")
    assert corrected == "ගුරුවරයා අපට උගන්වයි"

def test_torch_absent_fallback(monkeypatch):
    # Mock _HAS_TORCH to False to simulate environments without PyTorch installed
    import sinlib.spellcheck as sc
    monkeypatch.setattr(sc, "_HAS_TORCH", False)
    
    # Re-initialize detector
    detector = sc.TypoDetector()
    assert detector._bigru_detector is None
    assert detector.has_neural_labeler is False
    
    # Check that spelling checks and corrections still work with N-Gram model fallback
    assert not detector.is_word_suspicious("පාසලට")
    assert detector.is_word_suspicious("උගන්වය්")
    
    corrected = detector("ගුරුවරයා අපට උගන්වය්")
    assert corrected == "ගුරුවරයා අපට උගන්වයි"
