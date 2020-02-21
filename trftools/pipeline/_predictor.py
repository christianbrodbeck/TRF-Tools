# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path

from eelbrain import load, Dataset, NDVar, UTS, combine, epoch_impulse_predictor, event_impulse_predictor, resample
from eelbrain._experiment.definitions import typed_arg
import numpy

from .._ndvar import pad, shuffle
from ._code import NDVAR_SHUFFLE_METHODS, Code


class EventPredictor:
    """Generate an impulse for each epoch

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
        self.value = typed_arg(value, float, str)
        self.latency = typed_arg(latency, float, str)

    def _generate(self, uts: UTS, ds: Dataset, code: Code):
        assert code.stim is None
        return epoch_impulse_predictor((ds.n_cases, uts), self.value, self.latency, code.code, ds)

    def _generate_continuous(self, uts: UTS, ds: Dataset, code: Code):
        assert code.stim is None
        return event_impulse_predictor(uts, 'T_relative', self.value, self.latency, code.code, ds)


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
    refers to the predictor's name, and the optional ``options`` can be used to
    distinguish different sub-varieties of the same predictor.

    Predictors can be :class:`NDVar` (uniform time series) or :class:`Dataset`
    (non-uniform time series). When loading a predictor, :class:`Dataset`
    predictors are converted to :class:`NDVar` by placing impulses at
    time-stamps specified in the datasets. These datasets can contain the
    following columns:

     - ``time``: Time stamp of the event (impulse) in seconds.
     - ``value``: Value of the impulse (magnitude).
     - ``permute``: Boolean :class:`Var` indicating which cases/events should be
       permuted for ``$permute``. If missing, all cases are shuffled.
     - ``mask``: If present, the (boolean) mask will be applied to ``value``
       (``value`` will be set to zero wherever ``mask`` is ``False``).
       For ``$permute``, the mask is applied before permuting, and cases will be
       permuted only within the mask. For ``$remask``, ``mask`` is shuffled
       within the cases specified in ``permute``.
    """
    def __init__(self, resample=None):
        assert resample in (None, 'bin', 'resample')
        self.resample = resample

    def _load(self, tstep: float, filename: str, directory: Path):
        path = directory / f'{filename}.pickle'
        x = load.unpickle(path)
        # allow for pre-computed resampled versions
        if isinstance(x, list):
            xs = x
            for x in xs:
                if x.time.tstep == tstep:
                    break
            else:
                raise IOError(f"{path.name} does not contain tstep={uts.tstep!r}")
        elif isinstance(x, NDVar):
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
                raise RuntimeError(f"{path.name} has tstep={x.time.tstep}, not {tstep}")
            else:
                raise RuntimeError(f"resample={self.resample!r}")
        elif not isinstance(x, Dataset):
            raise TypeError(f'{x!r} at {path}')
        return x

    def _generate(self, uts: UTS, code: Code, directory: Path):
        x = self._load(uts.tstep, code.string_without_rand, directory)
        if isinstance(x, NDVar):
            x = pad(x, uts.tmin, nsamples=uts.nsamples)
        elif isinstance(x, Dataset):
            x = self._ds_to_ndvar(x, uts, code)
        else:
            raise RuntimeError(x)

        if code.shuffle in NDVAR_SHUFFLE_METHODS:
            x = shuffle(x, code.shuffle, code.shuffle_band, code.shuffle_angle)
            code.register_shuffle()
        return x

    def _generate_continuous(self, uts: UTS, ds: Dataset, stim_var: str, code: Code, directory: Path):
        cache = {stim: self._load(uts.tstep, code.with_stim(stim).string_without_rand, directory) for stim in ds[stim_var].cells}
        # determine type
        stim_type = {type(s) for s in cache.values()}
        assert len(stim_type) == 1
        stim_type = stim_type.pop()
        #
        if stim_type is Dataset:
            dss = []
            for t, stim in ds.zip('T_relative', stim_var):
                x = cache[stim].copy()
                x['time'] += t
                dss.append(x)
            x = self._ds_to_ndvar(combine(dss), uts, code)
        elif stim_type is NDVar:
            v = cache[ds[0, stim_var]]
            dimnames = v.get_dimnames(first='time')
            dims = (uts, *v.get_dims(dimnames[1:]))
            shape = [len(dim) for dim in dims]
            x = NDVar(numpy.zeros(shape), dims, code.code_with_rand)
            for t, stim in ds.zip('T_relative', stim_var):
                x_stim = cache[stim]
                i_start = uts._array_index(t + x_stim.time.tmin)
                i_stop = i_start + len(x_stim.time)
                x.x[i_start:i_stop] = x_stim.get_data(dimnames)
        return x

    @staticmethod
    def _ds_to_ndvar(ds: Dataset, uts: UTS, code: Code):
        if 'mask' in ds:
            mask = ds['mask'].x
            assert mask.dtype.kind == 'b', "'mask' must be boolean"
        else:
            mask = None

        if code.shuffle and 'permute' in ds:
            permute = ds['permute'].x
            assert permute.dtype.kind == 'b', "'permute' must be boolean"
            if code.shuffle == 'permute' and mask is not None:
                permute *= mask
        else:
            permute = None

        if code.shuffle == 'remask':
            if mask is None:
                raise code.error("$remask for predictor without mask", -1)
            rng = code._get_rng()
            if permute is None:
                rng.shuffle(mask)
            else:
                remask = mask[permute]
                rng.shuffle(remask)
                mask[permute] = remask
            code.register_shuffle()

        if mask is not None:
            ds['value'] *= mask

        if code.shuffle == 'permute':
            rng = code._get_rng()
            if permute is None:
                rng.shufffle(ds['value'].x)
            else:
                values = ds[permute, 'value'].x
                rng.shuffle(values)
                ds[permute, 'value'] = values
            code.register_shuffle()

        x = NDVar(numpy.zeros(len(uts)), uts, name=code.code_with_rand)
        ds = ds[ds['time'] < x.time.tstop]
        for t, v in ds.zip('time', 'value'):
            x[t] = v
        return x


class MakePredictor:
    """Predictor calls .make_predictor"""
