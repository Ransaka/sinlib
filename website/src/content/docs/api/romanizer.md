---
title: Romanizer
description: Full API reference for the sinlib Romanizer class.
---

Converts Sinhala text into Roman (Latin) script using phonological character mapping.

## Import

```python
from sinlib import Romanizer
```

## Quick Reference

| Method / Property | Returns | Description |
|---|---|---|
| `Romanizer()` | `Romanizer` | Initialize the romanizer |
| `romanizer(text)` | `str` or `list[str]` | Romanize a single text string or a batch of text strings |

## Usage

### Single Text

```python
from sinlib import Romanizer

romanizer = Romanizer()

romanizer("මම ගෙදර ගියා")
# 'mama gedara giya'

romanizer("ආයුබෝවන් සිංහල")
# 'ayubowan sinhala'
```

### Batch Romanization

```python
romanizer(["හෙලෝ", "වර්ල්ඩ්"])
# ['helo', 'warld']
```

### Mixed Content

Non-Sinhala characters, English words, numbers, and punctuation are preserved automatically:

```python
romanizer("සිංහල 123 english!")
# 'sinhala 123 english!'
```
