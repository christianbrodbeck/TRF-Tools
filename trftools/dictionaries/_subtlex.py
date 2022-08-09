# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import log
from typing import Literal

from ._utils import download


MINIMAL_ENTRY = {
    'FREQcount': 1,
    'CDcount': 1,
    'Lg10WF': log(2, 10),  # log10(FREQcount + 1)
    'Lg10CD': log(2, 10),
}
TOTAL_COUNT = 51e6
LANGUAGES = {
    'US': ('https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip/at_download/file', 'SUBTLEXus74286wordstextversion.txt'),
    'NL': ('http://crr.ugent.be/subtlex-nl/SUBTLEX-NL.txt.zip', 'SUBTLEX-NL.txt'),
}
KNOWN_DUPLICATES = {
    'US': (),
    'NL': ('wat', 'ged', 'scott', 'het', 'doen', 'wilhelmstrasse', 'scheisse'),
}


def read_subtlex(
        language: Literal['US', 'NL'] = 'US',
        lower: bool = False,
):
    """Read the SUBTLEXus data

    Parameters
    ----------
    language
        Language to load.
    lower
        Use lower case keys (default is upper case).

    Notes
    -----
    NL
        http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-nl
    US
        http://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus

    Columns:

    Word, FREQcount, CDcount, FREQlow, Cdlow, SUBTLWF, Lg10WF, SUBTLCD, Lg10CD

    """
    url, filename = LANGUAGES[language]
    path = download(url, filename, unzip=True)
    out = {}
    str_trans = str.lower if lower else str.upper
    known_duplicates = [str_trans(word) for word in KNOWN_DUPLICATES[language]]
    with path.open() as fid:
        columns = fid.readline().split()
        i_key = columns.index('Word')
        columns.pop(i_key)
        for line in fid:
            items = line.split('\t')
            key = str_trans(items.pop(i_key))
            if key in out and key not in known_duplicates:
                print(f"Duplicate key: {key}")
            out[key] = dict(zip(columns, map(float, items)))
    return out


def read_subtlex_pos(
        upper: bool = False,
):
    """Read SUBTLEXus with part-of-speech tags

    Parameters
    ----------
    upper
        Turn keys into all upper-case. By default, 'I' is the only upper-case
        key.

    Notes
    -----
    Possible POS tags:
    Adjective, Adverb, Article, Conjunction, Determiner, Ex, Interjection,
    Letter, Name, Not, Noun, Number, Preposition, Pronoun, To, Unclassified,
    Verb
    """
    path = download('http://crr.ugent.be/papers/SUBTLEX-US_frequency_list_with_PoS_information_final_text_version.zip', 'SUBTLEX-US-PoS.txt', unzip=True)
    with path.open() as fid:
        keys = next(fid).split()
        i_word = keys.index('Word')
        i_class = keys.index('All_PoS_SUBTLEX')
        i_freq = keys.index('All_freqs_SUBTLEX')
        rows = (line.split() for line in fid)
        subtlex = {row[i_word]: {k: int(v) for k, v in zip(row[i_class].split('.'), row[i_freq].split('.')) if v != '#N/A'} for row in rows}
    if upper:
        subtlex = {k.upper(): v for k, v in subtlex.items()}
    return subtlex
