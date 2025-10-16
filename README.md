# TRF-Tools
Tools for data analysis with multivariate temporal response functions (mTRFs).
This repository mostly contains tools that extend Eelbrain but are not deemed stable enough to be included in the main release.
Includes a TRF extension of the Eelbrain MneExperiment pipeline, documented [here](https://trf-tools.readthedocs.io/). 

## Installing

TRF-Tools should be installed into a recent 
[Eelbrain](https://eelbrain.readthedocs.io/) 
environment (see [Installing Eelbrain](https://eelbrain.readthedocs.io/en/stable/installing.html)).

For BIDS `Pipeline` functionality, use the following environment:

```Bash
$ mamba env create --file=https://github.com/christianbrodbeck/TRF-Tools/raw/bids/env-bids.yml
```

Then, activate the new environment:

```Bash
$ mamba activate bids
```

To later update TRF-Tools to the latest version of the `bids` branch, use:

```bash
$ pip install -U https://github.com/christianbrodbeck/TRF-Tools/archive/bids.zip
```

To see what version you have currently installed, run:

```bash
$ python -c "import trftools; print(trftools.__version__)"
```
