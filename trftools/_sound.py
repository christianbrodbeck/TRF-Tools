# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import NDVar, Scalar, UTS

from ._ndvar import pad as _pad_func


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
    sfreq = 1 / wav.time.tstep
    if tstep is None:
        tstep = wav.time.tstep
    x = gtgram(wav_.get_data('time'), sfreq, integration_window, tstep, n, f_min, f_max)
    freqs = centre_freqs(sfreq, n, f_min, f_max)
    # freqs = np.round(freqs, out=freqs).astype(int)
    freq_dim = Scalar('frequency', freqs[::-1], 'Hz')
    time_dim = UTS(tmin, tstep, x.shape[1])
    return NDVar(x, (freq_dim, time_dim), name or wav.name)
