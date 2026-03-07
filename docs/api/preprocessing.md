# Preprocessing Utilities

Low-level Sinhala text processing functions used internally by `Tokenizer`. The core algorithm in `process_text()` implements Sinhala-aware character splitting: it groups each base consonant together with any following vowel diacritics into a single phonological unit.

## Import

```python
from sinlib.utils.preprocessing import process_text, download_hub_file, Filenames
```

## Core Function

### `process_text`

```python
from sinlib.utils.preprocessing import process_text

process_text("ආයුබෝවන්")
# ['ආ', 'යු', 'බෝ', 'ව', 'න්']

process_text("සිංහල")
# ['සි', 'ං', 'හ', 'ල']
```

The function iterates over Unicode code points and attaches any vowel sign (category `Mc` or `Mn`) or virama (්) to the preceding consonant. This produces the phonologically meaningful units used by the tokenizer vocabulary.

!!! warning
    Do not modify `process_text()`. All downstream functionality (tokenizer, spell checker) depends on its exact output.

## File Download Utility

### `download_hub_file`

Downloads model artefacts from the HuggingFace Hub (`Ransaka/sinlib`) and caches them locally. Called automatically by `Tokenizer.from_pretrained()` and `TypoDetector`.

```python
from sinlib.utils.preprocessing import download_hub_file, Filenames

vocab_path = download_hub_file(Filenames.VOCAB.value)
```

### `Filenames` enum

| Member | Value | Description |
|---|---|---|
| `Filenames.VOCAB` | `"vocab.json"` | Token vocabulary |
| `Filenames.CHAR_MAP` | `"char_map.json"` | Character mapping |
| `Filenames.NGRAM_PROBS` | `"ngram_probs.npy"` | Bigram probabilities |
| `Filenames.DICTIONARY` | `"dictionary.npy"` | Word dictionary |

## API Reference

::: sinlib.utils.preprocessing
    options:
      show_source: true
      members:
        - process_text
        - download_hub_file
        - Filenames
