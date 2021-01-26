"""Carnegie Mellon University Pronouncing Dictionary

Doctionary r13272

http://www.speech.cs.cmu.edu/cgi-bin/cmudict?in=UNPLUCKED
"""
from collections import defaultdict
from pathlib import Path
import re

from ._arpabet import STRIP_STRESS_MAP
from ._utils import download


PUNC_WORD_SUB = {
    "NE'ER": 'NEVER',
}
IGNORE = (
    "AUJOURD'HUI",
    "AUJOURD'HUI'S",
    "B'GOSH",
    "B'NAI",
    "B'RITH",
    "BA'ATH",
    "BA'ATH",
    "BA'ATHISM",
    "BA'ATHIST",
    "BA'ATHISTS",
    "BAHA'IS",
    "BEL'C",
    "C'EST",
    "C'EST",
    "C'MON",
    "CARA'VERAS",
    "D'AGOSTINO'S",
    "D'ALENE'S",
    "D'ALESSANDRO'S",
    "D'AMATO'S",
    "D'AMORE'S",
    "D'ANGELO'S",
    "DELL'AQUILA",
    "DON'TS",
    "DON'TS",
    "G'VANNI'S",
    "HA'ARETZ",
    "HA'ARETZ",
    "HA'ETZNI",
    "HALLOWE'EN",
    "HORS_D'OEUVRE",
    "HORS_D'OEUVRES",
    "I'ERS",
    "M'BOW",
    "M'BOW",
    "MA'AM",
    "O'BRIEN'S",
    "O'CONNER'S",
    "O'CONNER'SO'DONNELL'S",
    "O'CONNOR'S",
    "O'DELL'S",
    "O'DONNELL'S",
    "O'GRADY'S",
    "O'HARA'S",
    "O'KEEFFE'S",
    "O'LEARY'S",
    "O'NEILL'S",
    "O'NUTS",
    "PAY'N",
    "ROCK'N'ROLL",
    "SHA'ATH",
    "T'ANG",
    "Y'ALL",
    "Y'KNOW",
)
PRE_FIXES = {
    'D': 'D',
    'L': 'L',
    'N': 'N',
    'O': ('AH0', 'OW0', 'OW1', 'OW2'),
}
POST_FIXES = {
    'D': 'D',
    'LL': 'L',
    'M': 'M',
    'RE': ('ER0', 'ER1', 'R'),
    'S': 'SZ',
    'T': 'T',
    'VE': 'V',
}


def read_cmupd(strip_stress=False, apostrophe="'"):
    """Read the CMU-Pronunciation Dictionary

    Parameters
    ----------
    strip_stress : bool
        Remove stress from pronunciations (default ``False``).
    apostrophe : str | bool
        Character to replace apostrophe with in keys (e.g., "COULDN'T"; default
        is to keep apostrophe; set to ``False`` to split entries with
        apostrophes into pre- and post-apostrophy components).

    Returns
    -------
    cmu : dict {str: list of str}
        Dictionary mapping words (all caps) to lists of pronunciations.
    """
    path = download('http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b', 'cmudict-0.7b.txt')
    out = defaultdict(set)
    for line in path.open('rb'):
        m = re.match(rb"^([\w']+)(?:\(\d\))?  ([\w ]+)$", line)
        if m:
            k, v = m.groups()
            out[k.decode()].add(v.decode())

    # remove apostrophes from keys
    if apostrophe != "'":
        keys = [key for key in out if "'" in key]
        if apostrophe is False:
            for key in keys:
                values = out.pop(key)
                # hard-coded exceptions
                if key in IGNORE:
                    continue
                elif key.count("'") > 1:
                    continue
                elif key in PUNC_WORD_SUB:
                    out[PUNC_WORD_SUB[key]].update(values)
                    continue

                a_index = key.index("'")
                # word-initial or -final apostrophy
                if a_index == 0 or a_index == len(key) - 1:
                    if a_index == 0:
                        key_a = key[1:]
                    else:
                        key_a = key[:-1]
                    out[key_a].update(values)
                    continue
                # word-medial apostrophy
                key_a, key_b = key.split("'")
                for value in values:
                    if key_b in POST_FIXES:
                        if key.endswith("N'T") and value.endswith("N"):
                            value_a = value
                            value_b = None
                        else:
                            value_a, value_b = value.rsplit(' ', 1)
                            assert value_b in POST_FIXES[key_b]
                    elif key_a in PRE_FIXES:
                        value_a, value_b = value.split(' ', 1)
                        assert value_a in PRE_FIXES[key_a]
                    else:
                        raise RuntimeError("    %r," % key)
                    for k, v in ((key_a, value_a), (key_b, value_b)):
                        if v is not None:
                            out[k].add(v)
        elif isinstance(apostrophe, str):
            for key in keys:
                out[key.replace("'", apostrophe)].update(out.pop(key))
        else:
            raise TypeError(f"apostrophe={apostrophe!r}")
    # remove stress from pronunciations
    if strip_stress:
        out = {word: {' '.join(STRIP_STRESS_MAP[p] for p in pronunciation.split())
                      for pronunciation in pronunciations}
               for word, pronunciations in out.items()}
    return out
