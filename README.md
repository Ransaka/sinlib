# Sinlib

<div align="center">

![Sinlib Logo](welcome.png)

[![PyPI version](https://badge.fury.io/py/sinlib.svg)](https://badge.fury.io/py/sinlib)
[![Python Versions](https://img.shields.io/pypi/pyversions/sinlib.svg)](https://pypi.org/project/sinlib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://sinlib.readthedocs.io)

A Python toolkit for Sinhala natural language processing — phonological tokenization, spell checking, and text preprocessing.

</div>

> **Note:** The `Romanizer` and `Transliterator` modules are temporarily unavailable due to a known bug and will be restored in a future release.

## Installation

```bash
pip install sinlib
```

## Quick Start

### Tokenization

```python
from sinlib import Tokenizer

tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

# Split into phonological units (base consonant + diacritics)
tokens = tokenizer.tokenize("ආයුබෝවන්")
# ['ආ', 'යු', 'බෝ', 'ව', 'න්']

# Encode to integer IDs
encoding = tokenizer("ආයුබෝවන්")
encoding.input_ids       # [4, 23, 18, 7, 12]
encoding.attention_mask  # [1, 1, 1, 1, 1]

# Batch encode with padding
batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
batch.input_ids  # [[4, 23, 18, 7, 12], [9, 31, 6, 0, 0]]
```

### Spell Checking

```python
from sinlib import TypoDetector

detector = TypoDetector.from_pretrained("Ransaka/sinlib")

# Auto-correct a sentence
detector("අපකරියට ගිය")
# 'අපකීර්තියට ගිය'

# Get correction suggestions
detector.suggest_correction("අඩිරාජ")
# ['අධිරාජ']
```

### Neural Typo Correction (optional, CharBERT)

TypoDetector can optionally delegate hard cases to a **Sinhala-CharBERT** neural
corrector — a dual-channel (subword + phonological akshara) seq2seq model. This
catches noise classes the statistical dictionary pipeline cannot fix: Singlish
transliteration, dialectal morphology (`යන්ඩ` → `යන්න`), ZWJ-damaged ligatures
(`ක්රීඩාව` → `ක්‍රීඩාව`), split/fused words, and Unicode decomposition errors.

Install the optional dependency and pick a backend mode:

```python
from sinlib import TypoDetector

# "denoise"  - bounded word-level neural fix when dictionary suggestions fail
# "seq2seq"  - open-vocabulary sentence-level fix when structural noise is detected
# "hybrid"   - both, in cascade (recommended)
detector = TypoDetector(neural_backend="hybrid")

detector("මම ගෙදර යන්ඩ ඕනේ")
# 'මම ගෙදර යන්න ඕනේ'

detector("මම gedara යන්න ඕනේ")
# 'මම ගෙදර යන්න ඕනේ'

detector("ක්රීඩාව")
# 'ක්‍රීඩාව'
```

| Kwarg | Default | Description |
|---|---|---|
| `neural_backend` | `None` | `None`, `"denoise"`, `"seq2seq"`, or `"hybrid"` |
| `backend_model` | `Ransaka/sinhala-charbert-seq2seq` | HF Hub repo id or local checkpoint dir (`pytorch_model.bin` + `char_vocab.json`) |
| `backend_device` | auto (cuda > mps > cpu) | Torch device |
| `backend_revision` | `None` | Pin a Hub revision |
| `backend_num_beams` | `4` | Beam width for generation |

Notes:

- **Default behavior is unchanged** — without `neural_backend` the detector is
  purely statistical and requires no torch.
- Neural corrections are gated: clean in-dictionary sentences never hit the
  model, and a candidate is accepted only if it scores no worse than the input
  (hallucination guard).
- If the checkpoint cannot be downloaded, the detector degrades gracefully to
  statistical-only correction with a warning.

### Preprocessing

```python
from sinlib import preprocessing

# Remove noise and normalise text
clean = preprocessing.process_text("Hello, මේ සිංහල වාක්‍යකි.")

# Compute Sinhala character ratio
ratio = preprocessing.get_sinhala_character_ratio(["මෙය සිංහල වාක්‍යක්"])
# [0.9]
```

## Why phonological tokenization?

Sinhala script combines a base consonant with one or more vowel diacritics into a single phonetic unit. Standard Unicode tokenization breaks these apart, producing incorrect representations for downstream tasks like ASR and TTS.

```
"ආයුබෝවන්"

Sinlib  →  ['ආ', 'යු', 'බෝ', 'ව', 'න්']   ✓ phonological units
Unicode →  ['ආ', 'ය', 'ු', 'බ', 'ෝ', 'ව', 'න', '්']   ✗ raw code points
```

Vocab and model weights are fetched automatically from [`Ransaka/sinlib`](https://huggingface.co/Ransaka/sinlib) on HuggingFace Hub at first use — no manual setup required.

## Documentation

Full documentation is available at **[sinlib.readthedocs.io](https://sinlib.readthedocs.io)**, including:

- [API Reference — Tokenizer](https://sinlib.readthedocs.io/en/latest/api/tokenizer/)
- [API Reference — TypoDetector](https://sinlib.readthedocs.io/en/latest/api/spellcheck/)
- [Guide: Tokenization](https://sinlib.readthedocs.io/en/latest/guides/tokenization/)
- [Guide: Spell Checking](https://sinlib.readthedocs.io/en/latest/guides/spellcheck/)

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/Ransaka/sinlib).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

MIT License — see the [LICENSE](LICENSE) file for details.
