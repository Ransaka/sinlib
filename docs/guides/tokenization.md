# Tokenization Guide

This guide explains how sinlib tokenizes Sinhala text, why it produces the output it does, and how to use every part of the `Tokenizer` API.

## Why Sinhala needs special tokenization

Sinhala script uses *combining diacritics*: a vowel sound is written as a mark attached to a base consonant. Unicode assigns separate code points to the base consonant and each diacritic, but linguistically they form a single phonological unit.

For example, <span class="sinhala-text" lang="si">ආයුබෝවන්</span> is eight Unicode code points:

| Code point | Char | Type |
|---|---|---|
| U+0D86 | <span class="sinhala-text" lang="si">ආ</span> | vowel letter |
| U+0D9A | <span class="sinhala-text" lang="si">ය</span> | consonant |
| U+0DD4 | <span class="sinhala-text" lang="si">ු</span> | vowel sign (attaches to ය) |
| U+0DB6 | <span class="sinhala-text" lang="si">බ</span> | consonant |
| U+0DDD | <span class="sinhala-text" lang="si">ෝ</span> | vowel sign (attaches to බ) |
| U+0DC0 | <span class="sinhala-text" lang="si">ව</span> | consonant |
| U+0DB1 | <span class="sinhala-text" lang="si">න</span> | consonant |
| U+0DCA | ් | virama (attaches to න) |

Splitting on code points gives `['ආ','ය','ු','බ','ෝ','ව','න','්']` — eight tokens that do not map to speech sounds. Sinlib instead produces `['ආ','යු','බෝ','ව','න්']` — five phonological units.

## Loading the tokenizer

```python
from sinlib import Tokenizer

# Default: loads vocab from HuggingFace Hub (Ransaka/sinlib)
tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

# Local directory (must contain vocab.json)
tokenizer = Tokenizer.from_pretrained("./my_tokenizer/")
```

## Tokenizing text

### `tokenize()` — strings only

```python
tokenizer.tokenize("ආයුබෝවන්")
# ['ආ', 'යු', 'බෝ', 'ව', 'න්']

tokenizer.tokenize("සිංහල")
# ['සි', 'ං', 'හ', 'ල']
```

### `encode()` — IDs only

```python
tokenizer.encode("ආයුබෝවන්")
# [4, 23, 18, 7, 12]
```

### `__call__` / `encode_plus()` — full `BatchEncoding`

```python
enc = tokenizer("ආයුබෝවන්")
enc.input_ids       # [4, 23, 18, 7, 12]
enc.attention_mask  # [1, 1, 1, 1, 1]
```

## Padding and truncation

```python
# Pad to a fixed length
enc = tokenizer("සිංහල", max_length=8, padding="max_length")
enc.input_ids
# [9, 31, 6, 29, 0, 0, 0, 0]  ← 0 is the pad token ID

# Truncate long sequences
enc = tokenizer("ආයුබෝවන්", max_length=3, truncation=True)
enc.input_ids
# [4, 23, 18]
```

## BOS / EOS tokens

```python
enc = tokenizer("සිංහල", add_special_tokens=True)
# Prepends BOS token ID and appends EOS token ID when configured
```

## Batch encoding

```python
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
batch.input_ids
# [[4, 23, 18, 7, 12], [9, 31, 6, 29, 0]]
#                               ↑ padded to match longest sequence
```

Or equivalently:

```python
batch = tokenizer.batch_encode(["ආයුබෝවන්", "සිංහල"], padding=True)
```

## Decoding

```python
tokenizer.decode([4, 23, 18, 7, 12])
# 'ආයුබෝවන්'

tokenizer.batch_decode([[4, 23, 18, 7, 12], [9, 31, 6, 29]])
# ['ආයුබෝවන්', 'සිංහල']
```

Skip special tokens (pad, BOS, EOS) during decoding:

```python
tokenizer.decode([1, 4, 23, 18, 7, 12, 2], skip_special_tokens=True)
# 'ආයුබෝවන්'
```

## Vocabulary

```python
vocab = tokenizer.get_vocab()
# {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3, 'ආ': 4, ...}

len(vocab)  # number of tokens

# Token ↔ ID conversion
tokenizer.convert_tokens_to_ids(['ආ', 'යු'])   # [4, 23]
tokenizer.convert_ids_to_tokens([4, 23])         # ['ආ', 'යු']
```

## Saving and loading locally

```python
# Save
tokenizer.save_pretrained("./my_tokenizer/")
# Writes ./my_tokenizer/vocab.json

# Reload later
tokenizer2 = Tokenizer.from_pretrained("./my_tokenizer/")
```

## Training on custom data

If the default vocabulary does not cover your corpus you can train a new tokenizer:

```python
corpus = ["සිංහල", "ආයුබෝවන්", ...]  # list of Sinhala strings

tokenizer = Tokenizer(model_max_length=64)
tokenizer.train(corpus)
tokenizer.save_pretrained("./custom_tokenizer/")
```

The training procedure calls `process_text()` on every string in `corpus` to collect the unique phonological units that form the vocabulary.

## Special tokens reference

| Token | Attribute | Default ID |
|---|---|---|
| Padding | `tokenizer.pad_token` | `0` |
| Unknown | `tokenizer.unk_token` | `1` |
| Beginning of sequence | `tokenizer.bos_token` | `2` |
| End of sequence | `tokenizer.eos_token` | `3` |

```python
tokenizer.pad_token        # '[PAD]'
tokenizer.pad_token_id     # 0
tokenizer.all_special_tokens
# ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
```
