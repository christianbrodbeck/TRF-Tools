# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import NDVar, Scalar, UTS
import eelbrain

import numpy as np
from ._ndvar import pad as _pad_func
from gammatone.filters import make_erb_filters, centre_freqs, erb_filterbank
import gammatone
from mne.filter import resample

def gammatone_bank(wav: NDVar, f_min: float, f_max: float, n: int, integration_window: float = 0.010, tstep: float = None, location: str = 'right', pad: bool = True, name: str = None) -> NDVar:
    """Gammatone filterbank response

    Parameters
    ----------
    wav : NDVar
        Sound input.
    f_min : scalar
        Lower frequency cutoff.
    f_max : scalar
        Upper frequency cutoff.
    n : int
        Number of filter channels.
    integration_window : scalar
        Integration time window in seconds (default 10 ms).
    tstep : scalar
        Time step size in the output (default is same as ``wav``).
    location : str
        Location of the output relative to the input time axis:

        - ``right``: gammatone sample at end of integration window (default)
        - ``left``: gammatone sample at beginning of integration window
        - ``center``: gammatone sample at center of integration window

        Since gammatone filter response depends on ``integration_window``, the
        filter response will be delayed relative to the analytic envlope. To
        ignore this delay, use `location='left'`
    pad : bool
        Pad output to match time axis of input.
    name : str
        NDVar name (default is ``wav.name``).

    Notes
    -----
    Requires the ``fmax`` branch of the gammatone library to be installed:

        $ pip install https://github.com/christianbrodbeck/gammatone/archive/fmax.zip
    """
    from gammatone.filters import centre_freqs
    from gammatone.gtgram import gtgram

    tmin = wav.time.tmin
    wav_ = wav
    if location == 'left':
        if pad:
            wav_ = _pad_func(wav, wav.time.tmin - integration_window)
    elif location == 'right':
        # tmin += window_time
        if pad:
            wav_ = _pad_func(wav, tstop=wav.time.tstop + integration_window)
    elif location == 'center':
        dt = integration_window / 2
        # tmin += dt
        if pad:
            wav_ = _pad_func(wav, wav.time.tmin - dt, wav.time.tstop + dt)
    else:
        raise ValueError(f"mode={location!r}")
    if tstep is None:
        tstep = wav.time.tstep
    elif tstep % wav.time.tstep:
        raise ValueError(f"tstep={tstep}: must be a multiple of wav tstep ({wav.time.tstep})")
    sfreq = 1 / wav.time.tstep


    # prevent out of memory error
    targetFs = 1000
    tstep = 1/targetFs

    downsampledLength = int(round(wav_.shape[0]/sfreq*targetFs))

    cfs = centre_freqs(sfreq, n, f_min, f_max)
    fullXe = np.zeros((n, downsampledLength))
    for rIdx, currentCfs in enumerate(cfs):
        fcoefs = np.flipud(make_erb_filters(sfreq, currentCfs))
        xf = erb_filterbank(wav_.get_data('time'), fcoefs)
        xe = np.power(xf, 2)

        # downsample to targetFs
        downsampledXe = resample(xe, down=sfreq/targetFs)
        fullXe[len(cfs) - 1 - rIdx, :] = downsampledXe

    nwin, hop_samples, ncols = gammatone.gtgram.gtgram_strides(targetFs, integration_window, tstep, fullXe.shape[1])
    x = np.zeros((n, ncols))

    for cnum in range(ncols):
        segment = fullXe[:, cnum * hop_samples + np.arange(nwin)]
        x[:, cnum] = np.sqrt(segment.mean(1))

    freqs = centre_freqs(targetFs, n, f_min, f_max)

    freq_dim = Scalar('frequency', freqs[::-1], 'Hz')
    time_dim = UTS(tmin, tstep, x.shape[1])

    xnew = np.nan_to_num(x)

    return NDVar(xnew, (freq_dim, time_dim), name = name or wav.name)

