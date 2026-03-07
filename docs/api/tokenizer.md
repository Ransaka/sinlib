# Tokenizer

Character-level Sinhala tokenizer with a HuggingFace-compatible API. The tokenizer splits Sinhala text into phonological units (base consonant + vowel diacritics) and maps them to integer IDs.

## Import

```python
from sinlib import Tokenizer
```

## Quick Reference

| Method | Returns | Description |
|---|---|---|
| `Tokenizer.from_pretrained(path)` | `Tokenizer` | Load from HF Hub or local directory |
| `tokenizer(text)` | `BatchEncoding` | Encode one or more texts |
| `tokenizer.tokenize(text)` | `List[str]` | Split text into token strings |
| `tokenizer.encode(text)` | `List[int]` | Encode text to ID list |
| `tokenizer.encode_plus(text)` | `BatchEncoding` | Encode with full metadata |
| `tokenizer.batch_encode(texts)` | `BatchEncoding` | Encode a list of texts |
| `tokenizer.batch_decode(ids)` | `List[str]` | Decode a batch of ID lists |
| `tokenizer.decode(ids)` | `str` | Decode a single ID list |
| `tokenizer.convert_tokens_to_ids(tokens)` | `List[int]` | Token strings → IDs |
| `tokenizer.convert_ids_to_tokens(ids)` | `List[str]` | IDs → token strings |
| `tokenizer.get_vocab()` | `Dict[str, int]` | Full vocabulary mapping |
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

The directory must contain a `vocab.json` file. Use `save_pretrained()` to create one.

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
encoding = tokenizer("ආයුබෝවන්", max_length=8, padding="max_length", truncation=True)
# input_ids padded to length 8
```

### Batch encoding

```python
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
batch.input_ids   # [[4, 23, 18, 7, 12], [9, 31, 6, 0, 0]]
```

## Special Tokens

| Attribute | Default | Description |
|---|---|---|
| `tokenizer.pad_token` | `"[PAD]"` | Padding token |
| `tokenizer.unk_token` | `"[UNK]"` | Unknown token |
| `tokenizer.bos_token` | `"[BOS]"` | Beginning of sequence |
| `tokenizer.eos_token` | `"[EOS]"` | End of sequence |

## Saving

```python
tokenizer.save_pretrained("./my_tokenizer/")
# Writes vocab.json to the directory
```

## API Reference

::: sinlib.tokenizer.Tokenizer
    options:
      show_source: true
      members:
        - from_pretrained
        - __call__
        - tokenize
        - encode
        - encode_plus
        - batch_encode
        - batch_decode
        - decode
        - convert_tokens_to_ids
        - convert_ids_to_tokens
        - get_vocab
        - all_special_tokens
        - save_pretrained
        - train
