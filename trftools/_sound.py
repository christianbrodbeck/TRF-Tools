# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import floor

from eelbrain import NDVar, Scalar, UTS
from eelbrain._utils import tqdm
from numba import njit, prange
import numpy as np

from ._ndvar import pad as _pad_func


@njit(parallel=True)
def aggregate(xf: np.ndarray, n_samples: int, step: float, n_window: int):
    xf **= 2
    out = np.empty(n_samples)
    for i in prange(n_samples):
        start = int(round(i * step))
        out[i] = xf[start: start + n_window].mean()
    return out


def gammatone_bank(
        wav: NDVar,
        f_min: float,
        f_max: float,
        n: int,
        integration_window: float = 0.010,
        tstep: float = None,
        location: str = 'right',
        pad: bool = True,
        name: str = None,
) -> NDVar:
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
    from gammatone.filters import centre_freqs, erb_filterbank
    from gammatone.gtgram import make_erb_filters

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
    fs = 1 / wav.time.tstep
    if tstep is None:
        tstep = wav.time.tstep
    wave = wav_.get_data('time')
    # based on gammatone library, rewritten to reduce memory footprint
    cfs = centre_freqs(fs, n, f_min, f_max)
    integration_window_len = int(round(integration_window * fs))
    output_n_samples = floor((len(wave) - integration_window_len) * wav.time.tstep / tstep)
    output_step = tstep / wav.time.tstep
    results = []
    for i, cf in tqdm(enumerate(reversed(cfs)), "Gammatone spectrogram", total=len(cfs), unit='band'):
        fcoefs = np.flipud(make_erb_filters(fs, cf))
        xf = erb_filterbank(wave, fcoefs)
        results.append(aggregate(xf[0], output_n_samples, output_step, integration_window_len))
    result = np.sqrt(results)
    # package output
    freq_dim = Scalar('frequency', cfs[::-1], 'Hz')
    time_dim = UTS(wav.time.tmin, tstep, output_n_samples)
    if name is None:
        name = wav.name
    return NDVar(result, (freq_dim, time_dim), name)
