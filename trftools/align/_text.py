# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import product
import re
import string


PUNC = set(string.punctuation + "\n\r")
PUNC.remove("'")
PUNC_WORD_SUB = {
    "NE'ER": 'NEVER',
}
TEXT_SUB = {
    '1': 'ONE',
    '2': 'TWO',
    '3': 'THREE',
    '4': 'FOUR',
    '5': 'FIVE',
    '6': 'SIX',
    '7': 'SEVEN',
    '8': 'EIGHT',
    '9': 'NINE',
    # Alternate spellings
    'BARQUE': 'BARK',
    'COLOUR': 'COLOR',
    'FAVOURED': 'FAVORED',
    'FAVOURITE': 'FAVORITE',
    'HARBOURED': 'HARBORED',
    'HONOUR': 'HONOR',
    'NEIGHBOURHOOD': 'NEIGHBORHOOD',
    'REALISE': 'REALIZE',
    'SAVOURY': 'SAVORY',
    'STRAITENED': 'STRAIGHTENED',
    'STRAITLY': 'STRAIGHTLY',
    'VERANDAHS': 'VERANDAS',
    'VIGOUR': 'VIGOR',
}
# Apostrophe
for pair in product(('I', 'YOU', 'HE', 'SHE', 'WE', 'THEY'), ('LL', 'D')):
    TEXT_SUB["'".join(pair)] = ' '.join(pair)
for neg in ('CAN', 'DON', 'ISN', 'AREN', 'WEREN', 'MUSTN'):
    TEXT_SUB[neg + "'T"] = neg + " T"


def text_to_words(text):
    """Transform text to word sequence required for aligner

    Parameters
    ----------
    text : str
        The text to transform.
    """
    ta = text.encode('ascii', 'replace').decode('utf-8')
    tu = ta.upper()
    # spelling variations
    for old, new in PUNC_WORD_SUB.items():
        tu = re.sub(r'\b%s\b' % old, new, tu)
    # punctuation
    for punc in PUNC:
        tu = tu.replace(punc, ' ')
    words = tu.split()
    words = [w.strip("'") for w in words]
    tu = ' '.join(words)
    tu = tu.replace("'S", ' S')
    # Substitutions
    for old, new in TEXT_SUB.items():
        tu = re.sub(r'\b%s\b' % old, new, tu)

    return tu.encode('ascii')
