# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import appdirs


def download(url: str, filename: str, unzip: bool = False):
    file_path = Path(appdirs.user_cache_dir('TRF-Tools')) / filename
    file_path.parent.mkdir(exist_ok=True)
    if unzip:
        filename, headers = urlretrieve(url)
        with ZipFile(filename) as zipfile:
            names = zipfile.namelist()
            assert len(names) == 1
            tmp_dst = file_path.parent / names.pop()
            zipfile.extract(tmp_dst.name, file_path.parent)
            tmp_dst.rename(file_path)
    else:
        urlretrieve(url, file_path)
    return file_path
