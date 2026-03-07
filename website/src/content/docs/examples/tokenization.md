---
title: Tokenization Example
description: End-to-end tokenization example with sinlib.
---

## Basic tokenization

```python
from sinlib import Tokenizer

tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

# Single string
tokens = tokenizer.tokenize("ආයුබෝවන්")
print(tokens)
# ['ආ', 'යු', 'බෝ', 'ව', 'න්']

enc = tokenizer("ආයුබෝවන්")
print(enc.input_ids)       # [4, 23, 18, 7, 12]
print(enc.attention_mask)  # [1, 1, 1, 1, 1]
```

## Batch encoding with padding

```python
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
print(batch.input_ids)
# [[4, 23, 18, 7, 12],
#  [9, 31,  6,  0,  0]]
```

## Round-trip encode → decode

```python
ids = tokenizer.encode("ආයුබෝවන්")
text = tokenizer.decode(ids)
print(text)  # 'ආයුබෝවන්'
```

## Save and reload

```python
tokenizer.save_pretrained("./my_tokenizer/")
t2 = Tokenizer.from_pretrained("./my_tokenizer/")
assert t2.encode("ආයුබෝවන්") == tokenizer.encode("ආයුබෝවන්")
```
