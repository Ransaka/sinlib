# import pytest
# from sinlib import Romanizer

# @pytest.fixture
# def romanizer():
#     return Romanizer()

# def test_single_romanization(romanizer):
#     sinhala_text = "මම ගෙදර ගියා"
#     result = romanizer(sinhala_text)
#     assert isinstance(result, str)
#     assert "mama" in result.lower()
#     assert " " in result  # Verify word boundaries

# def test_batch_romanization(romanizer):
#     texts = ["හෙලෝ", "වර්ල්ඩ්"]
#     results = romanizer(texts)
    
#     assert len(results) == 2
#     assert "helo" in results[0].lower()
#     assert "warld" in results[1].lower()

# def test_mixed_content(romanizer):
#     text = "සිංහල 123 english!"
#     result = romanizer(text)
#     assert "123" in result
#     assert "english" in result

# def test_empty_input(romanizer):
#     assert romanizer("") == ""
#     assert romanizer([]) == []

# def test_special_characters(romanizer):
#     text = "මේ සිංහල පරීක්ෂණයක්: සම්මිශ්‍රණයෙන්!"
#     result = romanizer(text)
#     assert ": " in result
#     assert "!" in result