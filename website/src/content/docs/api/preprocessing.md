---
title: Preprocessing
description: API reference for sinlib's low-level Sinhala text preprocessing utilities.
---

Low-level Sinhala text processing functions used internally by `Tokenizer`. The core algorithm in `process_text()` implements Sinhala-aware character splitting.

## Import

```python
from sinlib.utils.preprocessing import (
    process_text,
    download_hub_file,
    Filenames,
    remove_non_printable,
    remove_english_characters,
    get_sinhala_character_ratio,
)
```

## `process_text`

Splits a Sinhala string into phonological units by grouping each base consonant with any following vowel diacritics or virama.

```python
from sinlib.utils.preprocessing import process_text

process_text("ආයුබෝවන්")
# ['ආ', 'යු', 'බෝ', 'ව', 'න්']

process_text("සිංහල")
# ['සි', 'ං', 'හ', 'ල']
```

:::caution
Do not modify `process_text()`. All downstream functionality (tokenizer, spell checker) depends on its exact output.
:::

## `download_hub_file`

Downloads model artefacts from the HuggingFace Hub (`Ransaka/sinlib`) and caches them locally. Called automatically by `Tokenizer.from_pretrained()` and `TypoDetector`.

```python
from sinlib.utils.preprocessing import download_hub_file, Filenames

vocab_path = download_hub_file(Filenames.VOCAB.value)
```

## `Filenames` enum

| Member | Value | Description |
|---|---|---|
| `Filenames.VOCAB` | `"vocab.json"` | Token vocabulary |
| `Filenames.CONFIG` | `"config.json"` | Tokenizer configuration |
| `Filenames.CHAR_MAPPER` | `"char_map.json"` | Character mapping |
| `Filenames.NGRAM_PROBS` | `"ngram_probs.npy"` | Bigram probabilities |
| `Filenames.DICTIONARY` | `"dictionary.npy"` | Word dictionary |

## `remove_non_printable`

Removes non-printable characters from a string, keeping ASCII printable characters (U+0020–U+007E) and the Sinhala Unicode block (U+0D80–U+0DFF).

```python
from sinlib.utils.preprocessing import remove_non_printable

remove_non_printable("මම ගෙදර ගියා\x00")
# 'මම ගෙදර ගියා'
```

## `remove_english_characters`

Removes ASCII Latin characters (a–z / A–Z) from a text string.

```python
from sinlib.utils.preprocessing import remove_english_characters

remove_english_characters("Hello සිංහල World")
# 'සිංහල'
```

## `get_sinhala_character_ratio`

Calculates the ratio of Sinhala characters in a text string or a list of text strings.

```python
from sinlib.utils.preprocessing import get_sinhala_character_ratio

get_sinhala_character_ratio("මම ගෙදර ගියා.")
# 1.0

get_sinhala_character_ratio(["මම ගෙදර ගියා.", "This is an example."])
# [1.0, 0.0]
```
