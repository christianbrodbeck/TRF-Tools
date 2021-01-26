# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import log
from pathlib import Path

from ._utils import download


MINIMAL_ENTRY = {
    'FREQcount': 1,
    'CDcount': 1,
    'Lg10WF': log(2, 10),  # log10(FREQcount + 1)
    'Lg10CD': log(2, 10),
}
TOTAL_COUNT = 51e6


def read_subtlex(lower=False):
    """Read the SUBTLEXus data

    Parameters
    ----------
    lower : bool
        Use lower case keys (default is upper case).

    Notes
    -----
    http://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus

    Columns:

    Word, FREQcount, CDcount, FREQlow, Cdlow, SUBTLWF, Lg10WF, SUBTLCD, Lg10CD

    """
    path = download('https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip/at_download/file', 'SUBTLEXus74286wordstextversion.txt', unzip=True)
    out = {}
    str_trans = str.lower if lower else str.upper
    with path.open() as fid:
        columns = fid.readline().split()
        i_key = columns.index('Word')
        columns.pop(i_key)
        for line in fid:
            items = line.split()
            key = str_trans(items.pop(i_key))
            if key in out:
                raise RuntimeError(f"Duplicate key: {key}")
            out[key] = dict(zip(columns, map(float, items)))
    return out


def read_subtlex_pos():
    """Read SUBTLEXus with part-of-speech tags"""
    path = download('http://crr.ugent.be/papers/SUBTLEX-US_frequency_list_with_PoS_information_final_text_version.zip', 'SUBTLEX-US-PoS.txt', unzip=True)
    with path.open() as fid:
        keys = next(fid).split()
        i_word = keys.index('Word')
        i_class = keys.index('All_PoS_SUBTLEX')
        i_freq = keys.index('All_freqs_SUBTLEX')
        d = {}
        for line in fid:
            line = line.split()
            d[line[i_word]] = {k: int(v) for k, v in
                               zip(line[i_class].split('.'),
                                   line[i_freq].split('.')) if v != '#N/A'}
    return d
