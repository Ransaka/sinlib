---
title: Typo Correction Example
description: End-to-end spell checking and correction with sinlib's hybrid pipeline.
---

sinlib's `TypoDetector` uses a **three-stage hybrid pipeline** to detect and correct Sinhala spelling errors:

1. **Phonological Trie** — retrieves phonologically close dictionary candidates
2. **BiGRU Seq2Seq** — generates correction candidates from a neural encoder-decoder
3. **Stupid Backoff LM** — reranks candidates using a 151k-article news corpus bigram model

---

## Basic sentence correction

```python
from sinlib import TypoDetector

detector = TypoDetector.from_pretrained()

result = detector("ගුරුවරයා අපට උගන්වය්")
print(result)
# 'ගුරුවරයා අපට උගන්වයි'
```

## Diacritic errors

```python
result = detector("සිංහල බාෂාව ලස්සනයි")
print(result)
# 'සිංහල භාෂාව ලස්සනයි'
```


## Context-aware suggestions

Passing `prev_word` and `next_word` enables context-aware re-ranking. This does not use neural networks; instead, it uses a **statistical Bigram Language Model** compiled from a 151k-article news corpus evaluated using the **Stupid Backoff** algorithm. It promotes semantically and grammatically plausible candidate corrections over phonologically closer ones in specific contexts.

```python
# Without context — returns closest phonological match
detector.suggest_correction("ලසන")
# ['වසන', 'ලේඛන', ...]

# With context — promotes the semantically correct candidate
detector.suggest_correction(
    "ලසන",
    prev_word="ලංකාව",
    next_word="රටක්"
)
# ['ලස්සන', ...]
```

## Checking valid words

```python
# A correctly spelled word passes through unchanged
result = detector("මගේ ගෙදර ලස්සනයි")
print(result)
# 'මගේ ගෙදර ලස්සනයි'
```

## Detection without correction

```python
detector.is_word_suspicious("සිංහල")   # False — valid
detector.is_word_suspicious("සින්හල")  # True  — likely typo
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
# Stricter: flag more words as suspicious
strict = TypoDetector(threshold=1e-6)

# Lenient: only flag obvious typos
lenient = TypoDetector(threshold=1e-12)
```
