# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import defaultdict

from ._arpabet import STRIP_STRESS_MAP


APOSTROPHE_I = {
    'D': -1,
    'M': -1,
    'S': -1,
    'T': -1,
    'AM': -1,  # MA'AM -> M AE M
    'LL': -1,
    'RE': -1,
    'VE': -1,
}


def iter_dict(file_name=None):
    """Iterate through forced aligner dictionary

    Parameters
    ----------
    file_name : str
        Dictionary file name (optional, the default is the internal dictionary).

    Yields
    ------
    word : str
        The word.
    phonemes : str
        Phonemes, space-delimited
    """
    for line in open(file_name):
        line = line.strip()
        if line:
            yield line.split(None, 1)


def read_dict(file_name=None, strip_stress=False):
    """Read a forced aligner dictionary file

    Parameters
    ----------
    file_name : str
        Dictionary file name (optional, the default is the internal dictionary).
    strip_stress : bool
        Strip stress information from vowels (e.g., 'AH0' -> 'AH').

    Returns
    -------
    dictionary : dict {str: list of str}
        Dictionary mapping words (all caps) to lists of pronunciations.
    """
    out = defaultdict(set)
    for word, phonemes in iter_dict(file_name):
        if strip_stress:
            phonemes = ' '.join(STRIP_STRESS_MAP[p] for p in phonemes.split())
            # e.g., "HH IH0 Z" and "HH IH1 Z" both --> "HH IH Z"
        out[word].add(phonemes)
    return out


def combine_dicts(dicts):
    out = defaultdict(set)
    for d in dicts:
        for key, values in d.items():
            out[key].update(values)
    return out


def split_apostrophe(dic):
    "Split words with apostrophe into two words (To match SUBTLEX)"
    out = defaultdict(set)
    for key, values in dic.items():
        if "'" in key:
            if key.count("'") > 1:
                print(f"Skipping {key}")
                continue
            elif key.startswith("'"):
                key = key[1:]
            elif key.endswith("'"):
                key = key[:-1]
            else:
                key1, key2 = key.split("'")
                try:
                    split_i = APOSTROPHE_I[key2]
                except KeyError:
                    print(f'Ignoring {key2} ({key})')
                    continue
                out[key1].update(v[:split_i] for v in values)
                out[key2].update(v[split_i:] for v in values)
                continue
        out[key].update(values)
    return out
