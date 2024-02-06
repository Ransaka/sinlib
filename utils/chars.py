from string import punctuation

BASE_CONSONANTS = [
    'ක', 'ඛ', 'ග', 'ඝ', 'ඞ', 'ඟ',
    'ච', 'ඡ', 'ජ', 'ඣ', 'ඤ', 'ඦ',
    'ට', 'ඨ', 'ඩ', 'ඪ', 'ණ', 'ඬ',
    'ත', 'ථ', 'ද', 'ධ', 'න', 'ඳ',
    'ප', 'ඵ', 'බ', 'භ', 'ම', 'ඹ',
    'ය', 'ර', 'ල', 'ව',
    'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ',
]

SAN = [
    'ඟ', 'ඦ', 'ඬ', 'ඳ', 'ඹ'
]

SAN_MAPPING = {'ඟ': 'ංග', 'ඦ': 'ඤ්ජ', 'ඬ': 'ණ්ඩ', 'ඳ': 'න්ද', 'ඹ': 'ම්බ'}
REVERSE_SAN_MAPPING = {d: v for v, d in SAN_MAPPING.items()}

CONSONANTS = [c + '්' for c in BASE_CONSONANTS]

VOWELS = [
    'අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ',
    'ඍ', 'ඎ', 'එ', 'ඒ', 'ඓ', 'ඔ', 'ඕ', 'ඖ',
    'අං', 'අඃ',
]

VOWEL_DIACRITICS = [
    '', 'ා', 'ැ', 'ෑ', 'ි', 'ී', 'ු', 'ූ', 'ෘ',
    'ෲ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ',
    'ං', 'ඃ', '්'
]

LONG_TO_SHORT_VOWEL_DIACRITICS_MAPPING = {
    '': 'ා',
    'ෑ': 'ැ',
    'ී': 'ි',
    'ූ': 'ු',
    'ේ': 'ෙ',
    'ෝ': 'ො'
}

DIACRITICS_MAPPING = {v: d for v, d in zip(VOWELS, VOWEL_DIACRITICS)}

REVERSE_DIACRITICS_MAPPING = {d: v for v, d in zip(VOWELS, VOWEL_DIACRITICS)}

CONJUNCT_CONSONANTS = [
    'ක්ර', 'ඛ්ර', 'ග්ර', 'ඝ්ර', 'ඞ්ර', 'ඟ්ර',
    'ක්ය', 'ඛ්ය', 'ග්ය', 'ඝ්ය', 'ඞ්ය', 'ඟ්ය',
    'ක්ෂ', '෴',
]

NUMERALS = [
    '𑇡', '𑇢', '𑇣', '𑇤', '𑇥', '𑇦', '𑇧', '𑇨', '𑇩', '𑇪',
    '𑇫', '𑇬', '𑇭', '𑇮', '𑇯', '𑇰', '𑇱', '𑇲', '𑇳', '𑇴',
]


GOSHA_LETTERS = [
    'අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ',
    'ඍ', 'ඎ', 'එ', 'ඒ', 'ඓ', 'ඔ', 'ඕ', 'ඖ',
    'අං', 'අඃ',
    'ග', 'ඝ', 'ඞ',
    'ජ', 'ඣ', 'ඤ',
    'ඩ', 'ඪ', 'ණ',
    'ද', 'ධ', 'න',
    'බ', 'භ', 'ම',
    'ය', 'ර', 'ල', 'ව',
    'හ'
]

AGOSHA_LETTERS = [
    'ක්', 'ඛ්',
    'ච්', 'ඡ්',
    'ට්', 'ඨ්',
    'ත්', 'ථ්',
    'ප්', 'ඵ්',
]

AGOSHA_TO_GOSHA_MAPPING = {
    'ක්': 'ග්',
    'ඛ්': 'ඝ්',
    'ච්': 'ජ්',
    'ඡ්': 'ඣ්',
    'ට්': 'ඩ්',
    'ඨ්': 'ඪ්',
    'ත්': 'ද්',
    'ථ්': 'ධ්',
    'ප්': 'බ්',
    'ඵ්': 'භ්',
}
PUNKT = set(punctuation)
NUMBERS = set("1234567890")

NUBERS_AND_PUNKTS = PUNKT.union(NUMBERS)

ALL_LETTERS = VOWELS + BASE_CONSONANTS + AGOSHA_LETTERS + GOSHA_LETTERS
ALL_LETTERS = set(ALL_LETTERS)
