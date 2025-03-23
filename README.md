
# Sinlib

<div align="center">

![Sinlib Logo](sinlib.png)

[![PyPI version](https://badge.fury.io/py/sinlib.svg)](https://badge.fury.io/py/sinlib)
[![Python Versions](https://img.shields.io/pypi/pyversions/sinlib.svg)](https://pypi.org/project/sinlib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for Sinhala text processing and analysis
</div>

## Overview

Sinlib is a specialized Python library designed for processing and analyzing Sinhala text. It provides tools for tokenization, preprocessing, and romanization to facilitate natural language processing tasks for the Sinhala language.

## Features

- 🔤 **Tokenizer**: Advanced tokenization for Sinhala text
- 🔍 **Preprocessor**: Text preprocessing utilities including Sinhala character ratio analysis
- 🔄 **Romanizer**: Convert Sinhala text to Roman characters

## Installation

Install the latest stable version from PyPI:

```bash
pip install sinlib
```

## Usage Examples

### Tokenizer

Split Sinhala text into meaningful tokens:

```python
from sinlib import Tokenizer

# Sample Sinhala text
corpus = """මේ අතර, පෙබරවාරි මාසයේ පළමු දින 08 තුළ පමණක් විදෙස් සංචාරකයන් 60,122 දෙනෙකු මෙරටට පැමිණ තිබේ.
ඒ අනුව මේ වසරේ ගත වූ කාලය තුළ සංචාරකයන් 268‍,375 දෙනෙකු දිවයිනට පැමිණ ඇති බව සංචාරක සංවර්ධන අධිකාරිය සඳහන් කරයි.
ඉන් වැඩි ම සංචාරකයන් පිරිසක් ඉන්දියාවෙන් පැමිණ ඇති අතර, එම සංඛ්‍යාව 42,768කි.
ඊට අමතර ව රුසියාවෙන් සංචාරකයන් 39,914ක්, බ්‍රිතාන්‍යයෙන් 22,278ක් සහ ජර්මනියෙන් සංචාරකයන් 18,016 දෙනෙකු පැමිණ ඇති බව වාර්තා වේ."""

# Initialize and train the tokenizer
tokenizer = Tokenizer()
tokenizer.train([corpus])

# Encode text into tokens
encoding = tokenizer("මේ අතර, පෙබරවාරි මාසයේ පළමු")

# List tokens
tokens = [tokenizer.token_id_to_token_map[id] for id in encoding]
print(tokens)
# Output: ['මේ', ' ', 'අ', 'ත', 'ර', ',', ' ', 'පෙ', 'බ', 'ර', 'වා', 'රි', ' ', 'මා', 'ස', 'යේ', ' ', 'ප', 'ළ', 'මු']
```

### Preprocessor

Analyze Sinhala character ratio in text:

```python
from sinlib.preprocessing import get_sinhala_character_ratio

# Sample sentences with varying Sinhala content
sentences = [
    'මෙය සිංහල වාක්‍යක්',                                  # Full Sinhala
    'මෙය සිංහල වාක්‍යක් සමග english character කීපයක්',     # Mixed Sinhala and English
    'This is a complete English sentence'                   # Full English
]

# Calculate Sinhala character ratio for each sentence
ratios = get_sinhala_character_ratio(sentences)
print(ratios)
# Output: [0.9, 0.46875, 0.0]
```

### Romanizer

Convert Sinhala text to Roman characters:

```python
from sinlib import Romanizer

# Sample texts with Sinhala content
texts = [
    "hello, මේ මාසයේ ගත වූ දින 15ක කාලය තුළ කොළඹ නගරය ආශ්‍රිත ව",
    "මෑතකාලීන ව රට මුහුණ දුන් අභියෝගාත්මකම ආර්ථික කාරණාව ණය ප්‍රතිව්‍යුගතකරණය බව"
]

# Initialize the romanizer
romanizer = Romanizer(char_mapper_fp=None, tokenizer_vocab_path=None)

# Romanize the texts
romanized_texts = romanizer(texts)
print(romanized_texts)
# Output:
# ['hello, me masaye gatha wu dina 15ka kalaya thula kolaba nagaraya ashritha wa',
#  'methakaleena wa rata muhuna dun abhiyogathmakama arthika karanawa naya prathiwyugathakaranaya bawa']
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to all contributors who have helped with the development of Sinlib
- Special thanks to the Sinhala NLP community for their support and feedback
