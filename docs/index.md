---
hide:
  - navigation
  - toc
---

<div class="sinlib-hero">
  <div class="sinlib-hero__badge">v0.1.12 · MIT License</div>
  <h1 class="sinlib-hero__title">Sinlib</h1>
  <p class="sinlib-hero__subtitle">
    A Python toolkit for Sinhala NLP — phonological tokenization,
    n-gram spell checking, and preprocessing utilities.
  </p>
  <div class="sinlib-hero__actions">
    <a class="sinlib-btn sinlib-btn--primary" href="guides/tokenization/">
      Get Started
    </a>
    <a class="sinlib-btn sinlib-btn--ghost" href="https://github.com/Ransaka/sinlib">
      GitHub
    </a>
    <a class="sinlib-btn sinlib-btn--ghost" href="api/">
      API Reference
    </a>
  </div>
  <div class="sinlib-hero__install">
    <span class="prompt">$</span> pip install sinlib
  </div>
</div>

## What's inside

<div class="sinlib-cards">
  <div class="sinlib-card">
    <span class="sinlib-card__icon">🔤</span>
    <p class="sinlib-card__title">Phonological Tokenizer</p>
    <p class="sinlib-card__desc">Splits Sinhala text into base consonant + diacritic units rather than raw Unicode code points. HuggingFace-compatible API.</p>
  </div>
  <div class="sinlib-card">
    <span class="sinlib-card__icon">✅</span>
    <p class="sinlib-card__title">Spell Checker</p>
    <p class="sinlib-card__desc">N-gram language model backed by a ~45 000-word dictionary. Auto-corrects typos and suggests alternatives.</p>
  </div>
  <div class="sinlib-card">
    <span class="sinlib-card__icon">⚙️</span>
    <p class="sinlib-card__title">Preprocessing</p>
    <p class="sinlib-card__desc">Remove noise, compute Sinhala character ratios, and batch-process text with ready-made utilities.</p>
  </div>
  <div class="sinlib-card">
    <span class="sinlib-card__icon">☁️</span>
    <p class="sinlib-card__title">HuggingFace Hub</p>
    <p class="sinlib-card__desc">Vocab and model weights are fetched automatically from <code>Ransaka/sinlib</code> on first use — zero manual setup.</p>
  </div>
</div>

## Quick start

=== "Tokenization"

    ```python
    from sinlib import Tokenizer

    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    # Split into phonological units
    tokens = tokenizer.tokenize("ආයුබෝවන්")
    # ['ආ', 'යු', 'බෝ', 'ව', 'න්']

    # Encode to integer IDs
    encoding = tokenizer("ආයුබෝවන්")
    encoding.input_ids       # [4, 23, 18, 7, 12]
    encoding.attention_mask  # [1, 1, 1, 1, 1]
    ```

=== "Batch encoding"

    ```python
    from sinlib import Tokenizer

    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    batch = tokenizer(["ආයුබෝවන්", "සිංහල"], padding=True)
    batch.input_ids
    # [[4, 23, 18, 7, 12],
    #  [9, 31,  6, 0,  0]]   ← padded with 0
    ```

=== "Spell checking"

    ```python
    from sinlib import TypoDetector

    detector = TypoDetector.from_pretrained("Ransaka/sinlib")

    # Auto-correct a sentence
    detector("අපකරියට ගිය")
    # 'අපකීර්තියට ගිය'

    # Get suggestions
    detector.suggest_correction("අඩිරාජ")
    # ['අධිරාජ']
    ```

## Why phonological tokenization?

Sinhala combines a base consonant with one or more vowel diacritics into a single phonetic unit.
Standard Unicode tokenization breaks these apart, producing incorrect representations for downstream tasks like ASR and TTS.

<div class="sinhala-example">
<strong>ආයුබෝවන්</strong> → <code>['ආ', 'යු', 'බෝ', 'ව', 'න්']</code> &nbsp;&nbsp; ✓ phonological<br>
vs &nbsp; <code>['ආ', 'ය', 'ු', 'බ', 'ෝ', 'ව', 'න', '්']</code> &nbsp;&nbsp; ✗ raw Unicode
</div>

Sinlib implements this splitting natively and maps each unit to a stable integer ID.

## Explore the docs

| Section | Description |
|---|---|
| [API → Tokenizer](api/tokenizer.md) | Full method reference for `Tokenizer` |
| [API → TypoDetector](api/spellcheck.md) | Spell-checker reference |
| [API → BatchEncoding](api/encoding.md) | Return type reference |
| [API → Preprocessing](api/preprocessing.md) | Low-level text utilities |
| [Guide: Tokenization](guides/tokenization.md) | In-depth tokenization walkthrough |
| [Guide: Spell Checking](guides/spellcheck.md) | Spell checking walkthrough |

## Contributing & license

Contributions are welcome — open an issue or pull request on [GitHub](https://github.com/Ransaka/sinlib).
Released under the [MIT License](https://github.com/Ransaka/sinlib/blob/main/LICENSE).
