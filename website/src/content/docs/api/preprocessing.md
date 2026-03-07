---
title: Preprocessing
description: API reference for sinlib's low-level Sinhala text preprocessing utilities.
---

Low-level Sinhala text processing functions used internally by `Tokenizer`. The core algorithm in `process_text()` implements Sinhala-aware character splitting.

## Import

```python
from sinlib.utils.preprocessing import process_text, download_hub_file, Filenames
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
| `Filenames.CHAR_MAP` | `"char_map.json"` | Character mapping |
| `Filenames.NGRAM_PROBS` | `"ngram_probs.npy"` | Bigram probabilities |
| `Filenames.DICTIONARY` | `"dictionary.npy"` | Word dictionary |
