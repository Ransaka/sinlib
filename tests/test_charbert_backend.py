"""
Tests for the optional CharBERT neural backend integration in TypoDetector.

Network-free: the backend is mocked, and the vendored model code is exercised
with a tiny randomly-initialized configuration.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from sinlib.charbert.backend import CharBERTBackend
from sinlib.spellcheck import TypoDetector


@pytest.fixture
def mock_dictionary():
    return ["අම්මා", "තාත්තා", "මල්ලි", "අක්කා", "නංගි", "ගෙදර", "පාසල", "පොත", "බල්ලා", "පූසා"]


@pytest.fixture
def mock_ngram_probs():
    return {12: 0.1, 34: 0.2, 56: 0.05, 78: 0.15, 90: 0.3}


# ---------------------------------------------------------------------------
# Backend heuristics (no torch / no network required)
# ---------------------------------------------------------------------------


def test_structural_noise_latin():
    assert CharBERTBackend.has_structural_noise("mama gedara yanna one") is True
    assert CharBERTBackend.has_structural_noise("මම gedara යන්න ඕනේ") is True


def test_structural_noise_zwj_damage():
    # Virama directly followed by a consonant without ZWJ -> damaged ligature
    assert CharBERTBackend.has_structural_noise("\u0d9a\u0dca\u0dbb\u0dd3\u0da9\u0dcf") is True


@pytest.mark.parametrize(
    "text",
    [
        "මම ගෙදර යන්න ඕනේ",
        "ශ්\u200dරී ලංකාව",  # proper ZWJ ligature is NOT flagged
        "",
    ],
)
def test_structural_noise_negative(text):
    assert CharBERTBackend.has_structural_noise(text) is False


def test_invalid_neural_backend_value():
    with pytest.raises(ValueError):
        TypoDetector(neural_backend="bogus")


def test_default_has_no_neural_backend():
    detector = TypoDetector(lazy_loading=True)
    assert detector.neural_backend is None
    assert detector.has_charbert is False


# ---------------------------------------------------------------------------
# TypoDetector gating with a mocked CharBERT backend
# ---------------------------------------------------------------------------


class FakeCharBERTBackend:
    """Network-free stand-in for CharBERTBackend."""

    instances = []

    def __init__(self, model_id=None, device=None, revision=None, num_beams=4, **kw):
        self.model_id = model_id
        self.num_beams = num_beams
        self.sentence_calls = []
        self.word_calls = []
        type(self).instances.append(self)

    def _ensure_loaded(self):
        return True

    has_structural_noise = staticmethod(CharBERTBackend.has_structural_noise.__func__ if hasattr(CharBERTBackend.has_structural_noise, "__func__") else CharBERTBackend.has_structural_noise) if False else None

    @staticmethod
    def has_structural_noise(text):
        return CharBERTBackend.has_structural_noise(text)

    @property
    def is_available(self):
        return True

    def correct_sentence(self, text, num_beams=None, max_length=None):
        self.sentence_calls.append(text)
        return self._sentence_map.get(text, text)

    def correct_word(self, word, **kw):
        self.word_calls.append(word)
        return self._word_map.get(word, word)

    # Fixed mapping for deterministic tests
    _sentence_map = {
        "මම ගෙදර යන්ඩ ඕනේ": "මම ගෙදර යන්න ඕනේ",
    }
    _word_map = {}


@pytest.fixture
def neural_detector(mock_dictionary, mock_ngram_probs):
    """TypoDetector with neural_backend='seq2seq' and a mocked backend."""
    FakeCharBERTBackend.instances = []
    with patch("sinlib.spellcheck.download_hub_file") as mock_download:
        with patch("numpy.load") as mock_load:
            mock_load.side_effect = [np.array(mock_dictionary), mock_ngram_probs]
            with patch("sinlib.spellcheck.Tokenizer"):
                with patch("sinlib.spellcheck.CharBERTBackend", FakeCharBERTBackend):
                    with patch("sinlib.spellcheck.AksharaNGram"):
                        detector = TypoDetector(neural_backend="seq2seq")
                        # Deterministic akshara-ngram scoring
                        detector._akshara_ngram = MagicMock()
                        detector._akshara_ngram.score_word.return_value = -5.0
                        yield detector


def test_backend_loaded(neural_detector):
    assert neural_detector.neural_backend == "seq2seq"
    assert neural_detector.has_charbert is True
    assert len(FakeCharBERTBackend.instances) == 1


def test_neural_pass_triggers_on_structural_noise(neural_detector):
    # Latin-script input triggers the seq2seq gate
    result = neural_detector("mama gedara yanna one")
    backend = FakeCharBERTBackend.instances[0]
    assert len(backend.sentence_calls) == 1


def test_neural_pass_applies_correction(neural_detector):
    # 'යන්ඩ' is out-of-dictionary -> gate triggers; mapped fix applied
    result = neural_detector("මම ගෙදර යන්ඩ ඕනේ")
    assert result == "මම ගෙදර යන්න ඕනේ"


def test_clean_in_dictionary_sentence_unchanged(neural_detector):
    # Clean sentence: the model is consulted, but its (unchanged) output is
    # rejected by the acceptance guard, so the text is returned as-is.
    result = neural_detector("ගෙදර පොත")
    backend = FakeCharBERTBackend.instances[0]
    assert len(backend.sentence_calls) == 1
    assert result == "ගෙදර පොත"


def test_hallucination_only_punctuation_rejected(neural_detector):
    backend = FakeCharBERTBackend.instances[0]
    backend._sentence_map["මම ගෙදර යන්ඩ ඕනේ"] = "මම ගෙදර යන්ඩ ඕනේ\u2026"
    result = neural_detector("මම ගෙදර යන්ඩ ඕනේ")
    # Punctuation-only difference must be rejected
    assert "\u2026" not in result


def test_word_denoise_mode_fallback(mock_dictionary, mock_ngram_probs):
    """In 'denoise' mode, correct_word is consulted for unfixable words."""

    class WordBackend(FakeCharBERTBackend):
        _word_map = {"අම්ම්මාකො": "අම්මා"}

    WordBackend.instances = []
    with patch("sinlib.spellcheck.download_hub_file"), patch("numpy.load") as mock_load:
        mock_load.side_effect = [np.array(mock_dictionary), mock_ngram_probs]
        with patch("sinlib.spellcheck.Tokenizer"), patch(
            "sinlib.spellcheck.CharBERTBackend", WordBackend
        ), patch("sinlib.spellcheck.AksharaNGram"):
            detector = TypoDetector(neural_backend="denoise")
            detector._akshara_ngram = MagicMock()
            detector._akshara_ngram.score_word.return_value = -5.0
            # Force the statistical pipeline to flag the word and fail to fix it,
            # so the denoise hook runs.
            detector.is_word_suspicious = lambda w: True
            detector.suggest_correction = lambda *a, **k: ["No suggestion"]
            result = detector("අම්ම්මාකො")
            backend = WordBackend.instances[0]
            assert len(backend.word_calls) >= 1
            assert result == "අම්මා"


# ---------------------------------------------------------------------------
# Vendored model code: tiny random forward/generate smoke test
# ---------------------------------------------------------------------------


def test_tiny_model_generate():
    torch = pytest.importorskip("torch")
    from sinlib.charbert.config import SinhalaCharBERTConfig
    from sinlib.charbert.seq2seq_decoder import SinhalaCharBERTSeq2SeqModel

    torch.manual_seed(42)
    config = SinhalaCharBERTConfig(
        vocab_size=64,
        char_vocab_size=16,
        hidden_size=16,
        char_embedding_dim=8,
        char_gru_hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    model = SinhalaCharBERTSeq2SeqModel(config, num_decoder_layers=1, max_target_positions=16)
    model.eval()

    batch, m, n = 1, 5, 7
    input_ids = torch.randint(1, 64, (batch, m))
    char_input_ids = torch.randint(1, 16, (batch, n))
    start_char_idx = torch.zeros(batch, m, dtype=torch.long)
    end_char_idx = torch.full((batch, m), n - 1, dtype=torch.long)
    attention_mask = torch.ones(batch, m, dtype=torch.long)
    char_attention_mask = torch.ones(batch, n, dtype=torch.long)

    out_greedy = model.generate(
        input_ids=input_ids,
        char_input_ids=char_input_ids,
        start_char_idx=start_char_idx,
        end_char_idx=end_char_idx,
        attention_mask=attention_mask,
        char_attention_mask=char_attention_mask,
        max_length=8,
        bos_token_id=2,
        eos_token_id=3,
        num_beams=1,
    )
    assert out_greedy.shape[0] == batch
    assert out_greedy.shape[1] <= 9
    assert (out_greedy[:, 0] == 2).all()

    out_beam = model.generate(
        input_ids=input_ids,
        char_input_ids=char_input_ids,
        start_char_idx=start_char_idx,
        end_char_idx=end_char_idx,
        attention_mask=attention_mask,
        char_attention_mask=char_attention_mask,
        max_length=8,
        bos_token_id=2,
        eos_token_id=3,
        num_beams=2,
    )
    assert out_beam.shape[0] == batch
