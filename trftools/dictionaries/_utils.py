# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


def download(url: str, dst: Path, unzip: bool = False):
    dst.parent.mkdir(exist_ok=True)
    if unzip:
        filename, headers = urlretrieve(url)
        with ZipFile(filename) as zipfile:
            names = zipfile.namelist()
            assert len(names) == 1
            tmp_dst = dst.parent / names.pop()
            zipfile.extract(tmp_dst.name, dst.parent)
            tmp_dst.rename(dst)
    else:
        urlretrieve(url, dst)
