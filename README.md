# TRF-Tools
Tools for data analysis with multivariate temporal response functions (mTRFs).
This repository mostly contains tools that extend Eelbrain but are not deemed stable enough to be included in the main release.
Includes a TRF extension of the Eelbrain `Pipeline`, documented [here](https://trf-tools.readthedocs.io/).

## Installing

TRF-Tools works with the BIDS `Pipeline`, which is under active development.
For experiments using the old file structure, see the [legacy branch](https://github.com/christianbrodbeck/TRF-Tools/tree/legacy).

For setting up an environment for the BIDS TRF `Pipeline`, use the following Terminal commands:

```Bash
curl -L -O https://github.com/christianbrodbeck/TRF-Tools/raw/main/env-trf.yml
mamba env create -f env-trf.yml
```

Then, activate the new environment:

```Bash
mamba activate trf
```

To later update TRF-Tools to the latest version, use:

```bash
pip install -U https://github.com/christianbrodbeck/TRF-Tools/archive/main.zip
```

To see what version you have currently installed, run:

```bash
python -c "import trftools; print(trftools.__version__)"
```
