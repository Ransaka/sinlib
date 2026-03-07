# Sinlib

![Sinlib Logo](https://raw.githubusercontent.com/Ransaka/sinlib/refs/heads/main/welcome.png)

A Python toolkit for Sinhala natural language processing. Sinlib provides character-level tokenization tailored to Sinhala's phonological structure, plus spell-checking via an n-gram language model.

## Installation

```bash
pip install sinlib
```

## Quick Start

### Tokenization

```python
from sinlib import Tokenizer

tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

# Tokenize a word — returns phonological units, not raw characters
tokens = tokenizer.tokenize("ආයුබෝවන්")
# ['ආ', 'යු', 'බෝ', 'ව', 'න්']

# Encode to IDs (returns a BatchEncoding)
encoding = tokenizer("ආයුබෝවන්")
encoding.input_ids       # [4, 23, 18, 7, 12]
encoding.attention_mask  # [1, 1, 1, 1, 1]

# Batch encode
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
batch.input_ids          # [[4, 23, 18, 7, 12], [9, 31, 6, 0, 0]]
```

### Spell Checking

```python
from sinlib import TypoDetector

detector = TypoDetector.from_pretrained("Ransaka/sinlib")

detector("අපකරියට ගිය")
# 'අපකීර්තියට ගිය'

detector.suggest_correction("අඩිරාජ")
# ['අධිරාජ']
```

## Core Concepts

Sinhala combines a base consonant with one or more vowel diacritics into a single phonetic unit. For example,
<span class="sinhala-text" lang="si">ආයුබෝවන්</span> is meaningfully split as
<span class="sinhala-text" lang="si">[ආ, යු, බෝ, ව, න්]</span>
rather than individual Unicode code points. This representation is especially useful for ASR and TTS tasks.

Sinlib's tokenizer implements this splitting natively. All vocabulary and model weights are fetched automatically from [`Ransaka/sinlib`](https://huggingface.co/Ransaka/sinlib) on HuggingFace Hub at first use.

## Navigation

| Section | Description |
|---|---|
| [API Reference → Tokenizer](api/tokenizer.md) | Full `Tokenizer` method reference |
| [API Reference → BatchEncoding](api/encoding.md) | `BatchEncoding` return type |
| [API Reference → TypoDetector](api/spellcheck.md) | Spell-checker reference |
| [API Reference → Preprocessing](api/preprocessing.md) | Low-level text utilities |
| [Guide: Tokenization](guides/tokenization.md) | In-depth tokenization walkthrough |
| [Guide: Spell Checking](guides/spellcheck.md) | Spell checking walkthrough |

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/Ransaka/sinlib).

## License

MIT License. See the `LICENSE` file for details.
