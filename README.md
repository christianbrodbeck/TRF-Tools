# TRF-Tools
Tools for data analysis with multivariate temporal response functions (mTRFs). This repository mostly contains tools that extend Eelbrain but are not yet stable enough to be included in the main release.

## Installing

TRF-Tools should be installed into a recent [`eelbrain` environment](https://eelbrain.readthedocs.io/en/stable/installing.html), for example:

```Bash
$ conda create -n eelbrain -c conda-forge eelbrain
```

Then, activate the new environment and install TRF-Tools directly from GitHub:

```Bash
$ conda activate eelbrain
$ pip install https://github.com/christianbrodbeck/TRF-Tools/archive/main.zip
```

In order to use the `gammatone_bank` function, also install:

```bash
$ pip install https://github.com/christianbrodbeck/gammatone/archive/fmax.zip
```

To later update TRF-Tools to the latest version of the `main` branch, use:

```bash
$ pip install -U https://github.com/christianbrodbeck/TRF-Tools/archive/main.zip
```

Sometimes this will not actually replace the old installation (check `pip`'s output, it should say `Successfully installed trftools`). If it does not, uninstall the old version manually and try again:

```bash
$ pip uninstall trftools
```
