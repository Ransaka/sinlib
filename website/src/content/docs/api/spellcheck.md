---
title: TypoDetector
description: Full API reference for the sinlib TypoDetector spell checker.
---

Sinhala spell checker using a character-level n-gram language model combined with edit-distance candidate generation.

## Import

```python
from sinlib import TypoDetector
```

## Quick Reference

| Method / Property | Returns | Description |
|---|---|---|
| `TypoDetector.from_pretrained(repo)` | `TypoDetector` | Load from HF Hub |
| `detector(text)` | `str` | Correct a sentence |
| `detector.suggest_correction(word)` | `list[str]` | Closest dictionary matches |
| `detector.word_ngram_probability(word)` | `float` | N-gram likelihood score |
| `detector.get_dictionary()` | `set[str]` | Full word list |
| `detector.get_ngram_probs()` | `dict` | Full n-gram table |
| `detector.dictionary` | `str` | Human-readable summary |
| `detector.ngram_probs` | `str` | Human-readable summary |

## Usage

### Correct a sentence

```python
from sinlib import TypoDetector

detector = TypoDetector.from_pretrained("Ransaka/sinlib")

detector("අපකරියට ගිය")
# 'අපකීර්තියට ගිය'
```

### Get correction suggestions

```python
detector.suggest_correction("අඩිරාජ")
# ['අධිරාජ']

detector.suggest_correction("xyz")
# ['No suggestion']
```

### Score a word

```python
prob = detector.word_ngram_probability("සිංහල")
# 0.000032  (higher = more likely to be a real word)
```

### Inspect the dictionary

```python
print(detector.dictionary)
# Dictionary containing 45231 words.

words = detector.get_dictionary()
"ගෙදර" in words  # True
```

## Behaviour Details

For each word in the input sentence the detector:

1. Checks if the word is in the known dictionary — if yes, passes through unchanged.
2. Estimates the word's character-level bigram probability.
   - If `prob < threshold` (default `1e-8`): replaces with the top `suggest_correction` result.
   - If `threshold <= prob < 1.0`: emits a `UserWarning` but keeps the word.
3. On any processing error, emits a `UserWarning` and keeps the original word.
