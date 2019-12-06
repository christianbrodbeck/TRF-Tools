# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import ceil

from eelbrain import NDVar, UTS
from eelbrain._utils.numpy_utils import index
import numpy as np

SHUFFLE_METHODS = ('shift',)


def pad(ndvar, tstart=None, tstop=None, nsamples=None):
    """Pad (or crop) an NDVar in time

    Parameters
    ----------
    ndvar : NDVar
        NDVar to pad.
    tstart : float
        New tstart.
    tstop : float
        New tstop.
    nsamples : int
        New number of samples.
    """
    axis = ndvar.get_axis('time')
    time = ndvar.dims[axis]
    # start
    if tstart is None:
        if nsamples is not None:
            raise NotImplementedError("nsamples without tstart")
        n_add_start = 0
    elif tstart < time.tmin:
        n_add_start = int(ceil((time.tmin - tstart) / time.tstep))
    elif tstart > time.tmin:
        n_add_start = -time._array_index(tstart)
    else:
        n_add_start = 0

    # end
    if nsamples is None and tstop is None:
        n_add_end = 0
    elif nsamples is None:
        n_add_end = int((tstop - time.tstop) // time.tstep)
    elif tstop is None:
        n_add_end = nsamples - n_add_start - ndvar.time.nsamples
    else:
        raise TypeError("Can only specify one of tstart and nsamples")
    # need to pad?
    if not n_add_start and not n_add_end:
        return ndvar
    # construct padded data
    xs = [ndvar.x]
    shape = ndvar.x.shape
    # start
    if n_add_start > 0:
        shape_start = shape[:axis] + (n_add_start,) + shape[axis + 1:]
        xs.insert(0, np.zeros(shape_start))
    elif n_add_start < 0:
        xs[0] = xs[0][index(slice(-n_add_start, None), axis)]
    # end
    if n_add_end > 0:
        shape_end = shape[:axis] + (n_add_end,) + shape[axis + 1:]
        xs += (np.zeros(shape_end),)
    elif n_add_end < 0:
        xs[-1] = xs[-1][index(slice(None, n_add_end), axis)]
    x = np.concatenate(xs, axis)
    new_time = UTS(time.tmin - (time.tstep * n_add_start), time.tstep, x.shape[axis])
    return NDVar(x, ndvar.dims[:axis] + (new_time,) + ndvar.dims[axis + 1:], ndvar.name, ndvar.info)


def shuffle(ndvar, method, band=slice(None), angle=180):
    """Shuffle NDVar on the time axis

    Parameters
    ----------
    ndvar : NDVar  (..., time)
        NDVar with time axis.
    method : str
        Shuffling method (see notes).
    band : int | slice
        Index for which bands to shuffle (default is all bands).
    angle : int
        Angle by which to shift data.

    Notes
    -----
    Available methods:

    - ``shift``: split the NDVar and switch the first and the second parts.
    """
    if band is None:
        band = slice(None)

    if method not in SHUFFLE_METHODS:
        raise ValueError(f"method={method!r}; need one of {' | '.join(SHUFFLE_METHODS)}")
    elif method == 'shift':
        assert 0 <= angle < 360
        time_ax = ndvar.get_axis('time')
        assert 1 <= ndvar.ndim <= 2, f"ndvar must be 1d or 2d, got {ndvar!r}"
        i_mid = int(round(ndvar.x.shape[time_ax] * (angle / 360)))
        out = ndvar.copy()
        if ndvar.ndim == 1:
            out.x[:i_mid] = ndvar.x[-i_mid:]
            out.x[i_mid:] = ndvar.x[:-i_mid]
        elif time_ax == 1:
            out.x[band, :i_mid] = ndvar.x[band, -i_mid:]
            out.x[band, i_mid:] = ndvar.x[band, :-i_mid]
        elif time_ax == 0:
            out.x[:i_mid, band] = ndvar.x[-i_mid:, band]
            out.x[i_mid:, band] = ndvar.x[:-i_mid, band]
        else:
            raise ValueError(f"{ndvar}: More than 2d")
        return out
    else:
        raise RuntimeError(f"method={method!r}")
