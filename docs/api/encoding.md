# BatchEncoding

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

## API Reference

::: sinlib.encoding.BatchEncoding
    options:
      show_source: true
      members:
        - __init__
        - input_ids
        - attention_mask
        - to_dict
        - __getitem__
        - __repr__
        - __eq__
