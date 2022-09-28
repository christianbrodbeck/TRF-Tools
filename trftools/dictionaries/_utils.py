# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from typing import Dict, Set
from urllib.request import urlretrieve
from zipfile import ZipFile

import appdirs


def fix_apostrophe_pronounciations(
        dictionary: Dict[str, Set[str]],
) -> None:
    """Add affixes, remove any other words with apostrophe

    Make a pronunciation dictionary compatible with preprocessing in the
    ``speech`` module.
    """
    from ..align._textgrid import APOSTROPHE_TOKENS

    new = {
        "N'T": {'N T', 'AH N T'},
        "'D": {'D'},
        "'M": {'AH M', 'M'},
        "'S": {'AH Z', 'EH S', 'S', 'Z'},
        # "'T": {'T'},
        "'LL": {'L', 'AH L'},
        "'RE": {'R'},
        "'VE": {'V'},
    }
    keep = set(APOSTROPHE_TOKENS).union(new)
    remove = [key for key in dictionary if "'" in key and key not in keep]
    for key in remove:
        del dictionary[key]
    for word, pronunciations in new.items():
        dictionary[word].update(pronunciations)


def download(url: str, filename: str, unzip: bool = False):
    file_path = Path(appdirs.user_cache_dir('TRF-Tools')) / filename
    file_path.parent.mkdir(exist_ok=True)
    if unzip:
        filename, headers = urlretrieve(url)
        with ZipFile(filename) as zipfile:
            names = zipfile.namelist()
            assert len(names) == 1
            tmp_dst = file_path.parent / names[0]
            zipfile.extract(names[0], file_path.parent)
            tmp_dst.rename(file_path)
    else:
        urlretrieve(url, file_path)
    return file_path
