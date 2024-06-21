# Sinlib

![Alt text](sinlib.png)

## Installation

Using pypi
`pip install sinlib`

## Basic usage 

01. Tokenizer

```python
from sinlib import Tokenizer

corpus = """මේ අතර, පෙබරවාරි මාසයේ පළමු දින 08 තුළ පමණක් විදෙස් සංචාරකයන් 60,122 දෙනෙකු මෙරටට පැමිණ තිබේ.
ඒ අනුව මේ වසරේ ගත වූ කාලය තුළ සංචාරකයන් 268‍,375 දෙනෙකු දිවයිනට පැමිණ ඇති බව සංචාරක සංවර්ධන අධිකාරිය සඳහන් කරයි.
ඉන් වැඩි ම සංචාරකයන් පිරිසක් ඉන්දියාවෙන් පැමිණ ඇති අතර, එම සංඛ්‍යාව 42,768කි.
ඊට අමතර ව රුසියාවෙන් සංචාරකයන් 39,914ක්, බ්‍රිතාන්‍යයෙන් 22,278ක් සහ ජර්මනියෙන් සංචාරකයන් 18,016 දෙනෙකු පැමිණ ඇති බව වාර්තා වේ."""

tokenizer = Tokenizer()
tokenizer.train([corpus])

#encode text into tokens
encoding = tokenizer("මේ අතර, පෙබරවාරි මාසයේ පළමු")

#list tokens
[tokenizer.token_id_to_token_map[id] for id in encoding]
['මේ', ' ', 'අ', 'ත', 'ර', ',', ' ', 'පෙ', 'බ', 'ර', 'වා', 'රි', ' ', 'මා', 'ස', 'යේ', ' ', 'ප', 'ළ', 'මු']
```

02. Preprocessor
   ```python
sent = ['මෙය සිංහල වාක්‍යක්', 'මෙය සිංහල වාක්‍යක් සමග english character කීපයක්','This is complete english sentence']
print(sent)
#['මෙය සිංහල වාක්\u200dයක්', 'මෙය සිංහල වාක්\u200dයක් සමග english character කීපයක්', 'This is #complete english sentence']

from sinlib.preprocessing import get_sinhala_character_ratio

get_sinhala_character_ratio(sent)
#[0.9, 0.46875, 0.0]
```

03. Sinnhala Romanizer
   ```python
texts = ["hello, මේ මාසයේ ගත වූ දින 15ක කාලය තුළ කොළඹ නගරය ආශ්‍රිත ව", "මෑතකාලීන ව රට මුහුණ දුන් අභියෝගාත්මකම ආර්ථික කාරණාව ණය ප්‍රතිව්‍යුගතකරණය බව මුදල් රාජ්‍ය අමාත්‍ය ආචාර්ය රංජිත් සියඹ$$$ mahatha see more****"]

from sinlib import Romanizer

romanizer = Romanizer(char_mapper_fp = None, tokenizer_vocab_path = None)
romanizer(text)
#['hello, me masaye gatha wu dina 15ka kalaya thula kolaba nagaraya ashritha wa',
# 'methakaleena wa rata muhuna dun abhiyogathmakama arthika karanawa naya prathiwyugathakaranaya #bawa mudal rajya amathya acharya ranjith siyaba$$$ mahatha see more****']
```
