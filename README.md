# TRF-Tools
Tools for data analysis with multivariate temporal response functions (mTRFs).
This repository mostly contains tools that extend Eelbrain but are not deemed stable enough to be included in the main release.
Includes a TRF extension of the Eelbrain MneExperiment pipeline, documented [here](https://trf-tools.readthedocs.io/). 

## Installing

TRF-Tools should be installed into a recent 
[Eelbrain](https://eelbrain.readthedocs.io/) 
environment (see [Installing Eelbrain](https://eelbrain.readthedocs.io/en/stable/installing.html)),
for example:

```Bash
$ mamba env create --file=https://github.com/Eelbrain/Alice/raw/main/environment.yml
```

Then, activate the new environment and install TRF-Tools directly from GitHub:

```Bash
$ mamba activate eelbrain
$ pip install https://github.com/christianbrodbeck/TRF-Tools/archive/main.zip
```

To later update TRF-Tools to the latest version of the `main` branch, use:

```bash
$ pip install -U https://github.com/christianbrodbeck/TRF-Tools/archive/main.zip
```

To see what version you have currently installed, run:

```bash
$ python -c "import trftools; print(trftools.__version__)"
```
