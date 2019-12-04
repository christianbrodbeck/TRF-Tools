# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import defaultdict

from ._arpabet import STRIP_STRESS_MAP


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
