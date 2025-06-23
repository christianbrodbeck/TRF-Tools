# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Dict, Generator, Sequence, Union

from ._arpabet import STRIP_STRESS_MAP
from ._utils import download


PathArg = Union[Path, str]
# Index before which to split phonemes
APOSTROPHE_I = {
    'D': -1,  # YOU'D -> Y UW / D
    'M': -1,
    'S': -1,
    'T': -1,
    'AM': -1,  # MA'AM -> M AE M
    'LL': -1,
    'RE': -1,
    'VE': -1,
}
DICTS = {
    'english': 'https://github.com/MontrealCorpusTools/mfa-models/blob/master/dictionary/english.dict?raw=true',  # https://montreal-forced-aligner.readthedocs.io/en/latest/pretrained_models.html#available-pronunciation-dictionaries
}


def iter_dict(path: Path) -> Generator[str, str]:
    """Iterate through forced aligner dictionary

    Parameters
    ----------
    path
        Dictionary file name.

    Yields
    ------
    word
        The word.
    phonemes
        Phonemes, space-delimited
    """
    for line in open(path):
        line = line.strip()
        if line:
            yield line.split(None, 1)


def read_dict(
        dictionary: PathArg = 'english',
        strip_stress: bool = False,
        upper: bool = False,
) -> defaultdict:
    """Read a forced aligner dictionary file

    Parameters
    ----------
    dictionary
        Dictionary file name, or name of a built-in dictionary:

        - ``english``: Montreal Forced Aligner `English dictionary <https://montreal-forced-aligner.readthedocs.io/en/latest/pretrained_models.html#available-pronunciation-dictionaries>`_

    strip_stress
        Strip stress information from vowels (e.g., 'AH0' -> 'AH').
    upper
        Force keys to uppercase.

    Returns
    -------
    dictionary
        Dictionary mapping words to sets of pronunciations
        ``{str: {str, ...}}``.
    """
    if dictionary in DICTS:
        path = download(DICTS[dictionary], f'{dictionary}.dict')
    else:
        path = Path(dictionary)
    out = defaultdict(set)
    for word, phonemes in iter_dict(path):
        if upper:
            word = word.upper()
        if strip_stress:
            phonemes = ' '.join(STRIP_STRESS_MAP[p] for p in phonemes.split())
            # e.g., "HH IH0 Z" and "HH IH1 Z" both --> "HH IH Z"
        out[word].add(phonemes)
    return out


def combine_dicts(
        dicts: Sequence[Dict[Any, set]],
) -> Dict[str: set]:
    """Merge multiple dictionaries with :class:`set` values"""
    out = defaultdict(set)
    for d in dicts:
        for key, values in d.items():
            out[key].update(values)
    return out


def split_apostrophe(
        dic: Dict[str: set],
) -> Dict[str: set]:
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


def write_dict(
        dictionary: Dict[str, Collection[str]],
        file_name: PathArg,
        separator: str = '\t',
):
    """Write a pronunciation dictionary to a text file

    Parameters
    ----------
    dictionary
        Dictionary mapping words (all caps) to lists of pronunciations.
    file_name
        Destination file.
    separator
        String to separate words from their pronunciation. The Montral Forced
        Aligner expects tab (the default).
    """
    with open(file_name, 'w') as fid:
        for key in sorted(dictionary):
            for pronunciation in sorted(dictionary[key]):
                fid.write(f'{key}{separator}{pronunciation}\n')
