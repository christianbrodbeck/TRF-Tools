# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from urllib.request import urlretrieve


def download(url: str, dst: Path):
    dst.parent.mkdir(exist_ok=True)
    if url.endswith('.zip'):
        raise NotImplementedError(f"Please download the SUBTLEX data from {url}, extract the archive and move the file to {dst}")
    urlretrieve(url, dst)
