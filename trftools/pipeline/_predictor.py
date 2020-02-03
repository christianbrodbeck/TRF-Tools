# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from eelbrain import load, Dataset, NDVar, UTS, epoch_impulse_predictor, resample
import numpy

from .._ndvar import pad, shuffle
from ._code import NDVAR_SHUFFLE_METHODS


class EventPredictor:
    """Generate an impulse at each event

    Parameters
    ----------
    value : scalar | str
        Name of a :class:`Var` or :class:`Factor` in the events :class:`Dataset`
        (or expression resulting in one).
    latency : scalar | str
        Latency of the impulse relative to the event in seconds (or expression
        retrieving it from the events dataset).
    """

    def __init__(self, value=1., latency=0.):
        self.value = value
        self.latency = latency

    def _generate(self, time, ds, code):
        assert code.stim is None
        return epoch_impulse_predictor((ds.n_cases, time), self.value, self.latency, code.code, ds)


class FilePredictor:
    """Predictor stored in file(s)

    .. warning::
        When changing the file in which the predictor is stored, cached results
        using that predictor will not automatically be deleted. Use
        :meth:`TRFExperiment.invalidate` whenever replacing a predictors.

    Parameters
    ----------
    resample : 'bin' | 'resample'
        How to resample predictor. When analyses are done at different sampling
        rates, it is often convenient to generate predictors at a high sampling
        rate and then downsample dynamically to match the data.

         - ``bin``: averaging the values in time bins
         - ``resample``: use appropriate filter followed by decimation

        For predictors with non-continuous information, such as impulses,
        binning is more appropriate. Alternatively, the predictor can be saved
        as a list of :class:`NDVar` with all the needed sampling frequencies.

    Notes
    -----
    The file-predictor expects to find a file for each stimulus containing the
    predictor at::

        {root}/predictors/{stimulus}|{name}[-{options}].pickle

    Where ``stimulus`` refers to the name provided by ``stim_var``, ``name``
    refers to the predictor's name, and the optional ``options`` can define
    different sub-varieties of the same predictor.

    Predictors can be :class:`NDVar` (UTS) or :class:`Dataset` (NUTS). NUTS
    predictors should contain the following columns:

     - ``ttime``: Time stamp of the event/impulse
     - ``value``: Value of the impulse
     - ``permute``: Boolean :class:`Var` indicating which cases should be
       permuted for ``$permute``.
    """
    def __init__(self, resample=None):
        assert resample in (None, 'bin', 'resample')
        self.resample = resample

    def _load(self, path, tmin, tstep, n_samples, code, seed):
        x = load.unpickle(path)
        # allow for pre-computed resampled versions
        if isinstance(x, list):
            xs = x
            for x in xs:
                if x.time.tstep == tstep:
                    break
            else:
                raise IOError(f"{os.path.basename(path)} does not contain tstep={tstep!r}")
        # continuous UTS
        if isinstance(x, NDVar):
            if x.time.tstep == tstep:
                pass
            elif self.resample == 'bin':
                x = x.bin(tstep, label='start')
            elif self.resample == 'resample':
                srate = 1 / tstep
                int_srate = int(round(srate))
                srate = int_srate if abs(int_srate - srate) < .001 else srate
                x = resample(x, srate)
            elif self.resample is None:
                raise RuntimeError(f"{os.path.basename(path)} has tstep={x.time.tstep}, not {tstep}")
            else:
                raise RuntimeError(f"resample={self.resample!r}")
            x = pad(x, tmin, nsamples=n_samples)
        # NUTS
        elif isinstance(x, Dataset):
            ds = x
            if code.shuffle in ('permute', 'relocate'):
                rng = numpy.random.RandomState(seed)
                if code.shuffle == 'permute':
                    index = ds['permute'].x
                    assert index.dtype.kind == 'b'
                    values = ds[index, 'value'].x
                    rng.shuffle(values)
                    ds[index, 'value'] = values
                else:
                    rng.shuffle(ds['value'].x)
                code.register_shuffle()
            x = NDVar(numpy.zeros(n_samples), UTS(tmin, tstep, n_samples), name=code.code_with_rand)
            ds = ds[ds['time'] < x.time.tstop]
            for t, v in ds.zip('time', 'value'):
                x[t] = v
        else:
            raise TypeError(f'{x!r} at {path}')

        if code.shuffle in NDVAR_SHUFFLE_METHODS:
            x = shuffle(x, code.shuffle, code.shuffle_band, code.shuffle_angle)
            code.register_shuffle()
        return x


class MakePredictor:
    """Predictor calls .make_predictor"""
