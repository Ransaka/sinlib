import pytest
from sinlib import Transliterator


@pytest.fixture
def transliterator():
    return Transliterator()


def test_single_transliteration(transliterator):
    sinhala_text = "මම ගෙදර ගියා"
    result = transliterator(sinhala_text)
    assert isinstance(result, str)
    assert len(result) > 0


def test_batch_transliteration(transliterator):
    texts = ["හෙලෝ", "ගෙදර"]
    results = transliterator.batch_transliterate(texts)
    assert len(results) == 2
    assert isinstance(results[0], str)
    assert isinstance(results[1], str)


def test_empty_input(transliterator):
    assert transliterator("") == ""
    assert transliterator(None) == ""
    assert transliterator.batch_transliterate([]) == []
