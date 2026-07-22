---
title: Tokenizer
description: Full API reference for the sinlib Tokenizer class.
---

Character-level Sinhala tokenizer with a HuggingFace-compatible API. Splits Sinhala text into phonological units (base consonant + vowel diacritics) and maps them to integer IDs.

## Import

```python
from sinlib import Tokenizer
```

## Quick Reference

| Method | Returns | Description |
|---|---|---|
| `Tokenizer.from_pretrained(path)` | `Tokenizer` | Load from HF Hub or local directory |
| `tokenizer(text)` | `BatchEncoding` | Encode one or more texts |
| `tokenizer.tokenize(text)` | `list[str]` | Split text into token strings |
| `tokenizer.encode(text)` | `list[int]` | Encode text to ID list |
| `tokenizer.encode_plus(text)` | `BatchEncoding` | Encode with full metadata |
| `tokenizer.batch_encode(texts)` | `BatchEncoding` | Encode a list of texts |
| `tokenizer.batch_decode(ids)` | `list[str]` | Decode a batch of ID lists |
| `tokenizer.decode(ids)` | `str` | Decode a single ID list |
| `tokenizer.convert_tokens_to_ids(tokens)` | `list[int]` | Token strings → IDs |
| `tokenizer.convert_ids_to_tokens(ids)` | `list[str]` | IDs → token strings |
| `tokenizer.get_vocab()` | `dict[str, int]` | Full vocabulary mapping |
| `tokenizer.save_pretrained(path)` | `None` | Save vocab to directory |

## Loading

### From HuggingFace Hub

```python
tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
```

### From a local directory

```python
tokenizer = Tokenizer.from_pretrained("./my_tokenizer/")
```

The directory must contain `vocab.json` and `config.json` files. Use `save_pretrained()` to create them.

### Legacy (deprecated)

```python
tokenizer = Tokenizer(max_length=16)
tokenizer.load_from_pretrained(load_default_tokenizer=True)  # DeprecationWarning
```

## Encoding

### Single text

```python
encoding = tokenizer("ආයුබෝවන්")
# BatchEncoding(input_ids=[4, 23, 18, 7, 12], attention_mask=[1, 1, 1, 1, 1])
```

### With padding and truncation

```python
encoding = tokenizer("ආයුබෝවන්", max_length=8, padding=True, truncation=True)
```

### Batch encoding

```python
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
batch.input_ids   # [[4, 23, 18, 7, 12], [9, 31, 6, 0, 0]]
```

## Special Tokens

| Attribute | Default | Description |
|---|---|---|
| `tokenizer.pad_token` | `"<|pad|>"` | Padding token |
| `tokenizer.unk_token` | `"<|unk|>"` | Unknown token |
| `tokenizer.bos_token` | `"<|bos|>"` | Beginning of sequence |
| `tokenizer.eos_token` | `"<|end_of_text|>"` | End of sequence |

## Saving

```python
tokenizer.save_pretrained("./my_tokenizer/")
# Writes vocab.json and config.json to the directory
```
