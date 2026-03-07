---
title: BatchEncoding
description: API reference for the BatchEncoding return type.
---

`BatchEncoding` is the return type of `Tokenizer.__call__`, `Tokenizer.encode_plus()`, and `Tokenizer.batch_encode()`. It provides both attribute-style and dict-style access to tokenizer output.

## Import

```python
from sinlib import BatchEncoding
# or
from sinlib.encoding import BatchEncoding
```

## Overview

```python
from sinlib import Tokenizer

tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
encoding = tokenizer("ආයුබෝවන්")

# Attribute access
encoding.input_ids       # [4, 23, 18, 7, 12]
encoding.attention_mask  # [1, 1, 1, 1, 1]

# Dict-style access
encoding["input_ids"]
encoding["attention_mask"]

# Convert to plain dict
encoding.to_dict()
# {"input_ids": [4, 23, 18, 7, 12], "attention_mask": [1, 1, 1, 1, 1]}
```

Batch encoding returns a `BatchEncoding` where each field is a list of lists:

```python
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
batch.input_ids   # [[4, 23, 18, 7, 12], [9, 31, 6, 0, 0]]
```

## Fields

| Field | Type | Description |
|---|---|---|
| `input_ids` | `list[int]` or `list[list[int]]` | Token IDs |
| `attention_mask` | `list[int]` or `list[list[int]]` | `1` for real tokens, `0` for padding |

## Methods

| Method | Description |
|---|---|
| `to_dict()` | Returns a plain `dict` with all fields |
| `__getitem__(key)` | Dict-style access |
| `__repr__()` | Human-readable representation |
| `__eq__(other)` | Equality comparison |
