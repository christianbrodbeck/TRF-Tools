# TRF-Tools
Tools for data analysis with temporal response functions (TRFs). This repository mostly contains tools that extend Eelbrain but are not deemed stable enough to be included in the release.

## Installing

TRF-Tools should be installed with `pip` into a recent [`eelbrain` environment](https://github.com/christianbrodbeck/Eelbrain/wiki/Installing). To install directly from GitHub:

```Bash
$ pip install https://github.com/christianbrodbeck/TRF-Tools/archive/master.zip
```

In order to use the `gammatone_bank` function, also install:

```bash
$ pip install https://github.com/christianbrodbeck/gammatone/archive/fmax.zip
```

To later update TRF-Tools to the latest version of the master branch, use:

```bash
$ pip install -U https://github.com/christianbrodbeck/TRF-Tools/archive/master.zip --no-cache-dir
```

### Windows

The TRF-Tools `master` branch is currently incompatible with the Windows operating system due to certain restrictions on filenames under Windows. If you are using Windows, use the `win` branch instead:

```Bash
$ pip install -U https://github.com/christianbrodbeck/TRF-Tools/archive/win.zip
```
