---
title: Typo Correction Example
description: End-to-end spell checking example with sinlib.
---

## Basic correction

```python
from sinlib import TypoDetector

detector = TypoDetector.from_pretrained("Ransaka/sinlib")

result = detector("අපකරියට ගිය")
print(result)
# 'අපකීර්තියට ගිය'
```

## Suggestions

```python
suggestions = detector.suggest_correction("අඩිරාජ")
print(suggestions)
# ['අධිරාජ']
```

## Checking valid words

```python
# A correctly spelled word passes through unchanged
result = detector("මගේ ගෙදර ලස්සනයි")
print(result)
# 'මගේ ගෙදර ලස්සනයි'
```

## Scoring words manually

```python
prob = detector.word_ngram_probability("සිංහල")
print(prob)
# ~3.2e-05  — plausible word

prob = detector.word_ngram_probability("xzqabc")
print(prob)
# ~1e-27  — implausible, would be corrected
```

## Tuning threshold

```python
# Stricter: flag more words
strict = TypoDetector(threshold=1e-6)

# Lenient: only obvious typos
lenient = TypoDetector(threshold=1e-12)
```
