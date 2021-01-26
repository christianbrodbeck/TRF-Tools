"""
Setuptools bootstrap module:
https://setuptools.readthedocs.io

Writing setup.py:
https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
"""

from distutils.version import LooseVersion
import re
from setuptools import setup, find_packages


# version must be in X.X.X format, e.g., "0.0.3dev"
with open('trftools/__init__.py') as fid:
    text = fid.read()
match = re.search(r"__version__ = '([.\w]+)'", text)
if match is None:
    raise ValueError("No valid version string found in:\n\n" + text)
version = match.group(1)
LooseVersion(version)  # check that it's a valid version

setup(
    name='trftools',
    version=version,
    author="Christian Brodbeck",
    author_email='christianbrodbeck@me.com',
    description="Tools for data analysis with temporal response functions.",
    keywords="MEG EEG Eelbrain",
    url="https://github.com/christianbrodbeck/TRF-Tools",
    install_requires=[
        "textgrid >=1.5",
        "filelock",
        "appdirs",
    ],
    packages=find_packages(),
    entry_points={
        'gui_scripts': (
            'trf-tools-make-jobs = trftools.pipeline._jobs:make_jobs_command',
        ),
    }
)
