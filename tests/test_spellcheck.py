import pytest
import warnings
import numpy as np
from unittest.mock import patch, MagicMock
from sinlib.spellcheck import TypoDetector


@pytest.fixture
def mock_dictionary():
    return ["අම්මා", "තාත්තා", "මල්ලි", "අක්කා", "නංගි", "ගෙදර", "පාසල", "පොත", "බල්ලා", "පූසා"]


@pytest.fixture
def mock_ngram_probs():
    # Mock n-gram probabilities for testing
    return {
        12: 0.1,  # Example n-gram key
        34: 0.2,
        56: 0.05,
        78: 0.15,
        90: 0.3
    }


@pytest.fixture
def mock_typo_detector(mock_dictionary, mock_ngram_probs):
    with patch('sinlib.spellcheck.download_hub_file') as mock_download:
        with patch('numpy.load') as mock_load:
            # Configure the mocks
            mock_load.side_effect = [
                np.array(mock_dictionary),
                mock_ngram_probs
            ]
            
            # Mock tokenizer
            with patch('sinlib.spellcheck.Tokenizer') as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value.load_from_pretrained.return_value = mock_tokenizer
                
                # Configure tokenizer to return simple token IDs
                def mock_tokenize(word, truncate_and_pad=False):
                    return [ord(c) for c in word]
                
                mock_tokenizer.side_effect = mock_tokenize
                
                detector = TypoDetector()
                return detector


def test_initialization(mock_typo_detector):
    """Test that TypoDetector initializes correctly."""
    assert mock_typo_detector is not None
    assert hasattr(mock_typo_detector, '_dictionary')
    assert hasattr(mock_typo_detector, '_ngram_probs')
    assert hasattr(mock_typo_detector, '_tokenizer')


def test_dictionary_property(mock_typo_detector, mock_dictionary):
    """Test the dictionary property returns the expected description."""
    description = mock_typo_detector.dictionary
    assert isinstance(description, str)
    assert str(len(mock_dictionary)) in description
    assert "Dictionary containing" in description


def test_get_dictionary(mock_typo_detector, mock_dictionary):
    """Test get_dictionary returns the full dictionary."""
    dictionary = mock_typo_detector.get_dictionary()
    assert isinstance(dictionary, set)
    assert len(dictionary) == len(mock_dictionary)
    assert set(dictionary) == set(mock_dictionary)


def test_ngram_probs_property(mock_typo_detector, mock_ngram_probs):
    """Test the ngram_probs property returns the expected description."""
    description = mock_typo_detector.ngram_probs
    assert isinstance(description, str)
    assert str(len(mock_ngram_probs)) in description
    assert "N-gram probability dictionary" in description


def test_get_ngram_probs(mock_typo_detector, mock_ngram_probs):
    """Test get_ngram_probs returns the full n-gram probabilities."""
    probs = mock_typo_detector.get_ngram_probs()
    assert isinstance(probs, dict)
    assert len(probs) == len(mock_ngram_probs)
    assert set(probs.keys()) == set(mock_ngram_probs.keys())


def test_word_ngram_probability(mock_typo_detector):
    """Test word_ngram_probability calculates probabilities correctly."""
    # Configure the mock tokenizer to return specific values
    mock_typo_detector._tokenizer = lambda word, truncate_and_pad: [1, 2, 3, 4]
    
    # Mock the n-gram probabilities
    mock_typo_detector._ngram_probs = {12: 0.5, 23: 0.4, 34: 0.3}
    
    # Calculate probability with n=2
    prob = mock_typo_detector.word_ngram_probability("test", n=2)
    
    # Expected: 0.5 * 0.4 * 0.3 = 0.06
    assert prob == pytest.approx(0.06)


def test_suggest_correction(mock_typo_detector, mock_dictionary):
    """Test suggest_correction returns appropriate suggestions."""
    # Test with a word similar to one in the dictionary
    with patch('sinlib.spellcheck.get_close_matches') as mock_get_close:
        mock_get_close.return_value = ["අම්මා"]
        suggestions = mock_typo_detector.suggest_correction("අම්ම")
        assert suggestions == ["අම්මා"]
        mock_get_close.assert_called_once()
    
    # Test with a word that has no close matches
    with patch('sinlib.spellcheck.get_close_matches') as mock_get_close:
        mock_get_close.return_value = []
        suggestions = mock_typo_detector.suggest_correction("xyz")
        assert suggestions == ["No suggestion"]


def test_check_spelling_correct_word(mock_typo_detector):
    """Test check_spelling with a correctly spelled word."""
    mock_typo_detector._dictionary = ["correct"]
    result = mock_typo_detector("correct")
    assert result == "correct"


def test_check_spelling_typo(mock_typo_detector):
    """Test check_spelling with a typo."""
    mock_typo_detector._dictionary = ["correct"]
    
    # Mock word_ngram_probability to return a low probability
    mock_typo_detector.word_ngram_probability = lambda word, n=2: 1e-10
    
    # Mock suggest_correction
    mock_typo_detector.suggest_correction = lambda word, n=3: ["correct"]
    
    result = mock_typo_detector("incorrekt")
    assert result == "correct"


def test_check_spelling_unusual_word(mock_typo_detector):
    """Test check_spelling with an unusual but possibly valid word."""
    mock_typo_detector._dictionary = ["common"]
    
    # Mock word_ngram_probability to return a medium probability
    mock_typo_detector.word_ngram_probability = lambda word, n=2: 1e-7
    
    with warnings.catch_warnings(record=True) as w:
        result = mock_typo_detector("uncommon")
        assert result == "uncommon"
        assert len(w) == 1
        assert "unusual but may not be a typo" in str(w[0].message)