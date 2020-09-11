# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import chain
from pathlib import Path

from eelbrain import load, Categorial, Dataset, Factor, NDVar, UTS, Var, combine, epoch_impulse_predictor, event_impulse_predictor, resample
from eelbrain._experiment.definitions import typed_arg
import numpy

from .._ndvar import pad, shuffle
from ._code import NDVAR_SHUFFLE_METHODS, Code


def t_stop_ds(ds: Dataset, t: float):
    "Dummy-event for the end of the last step"
    t_stop = ds.info['tstop'] + t
    out = {}
    for k, v in ds.items():
        if k == 'time':
            out['time'] = Var([t_stop])
        elif isinstance(v, Var):
            out[k] = Var(numpy.asarray([0], v.x.dtype))
        elif isinstance(v, Factor):
            out[k] = Factor([''])
        else:
            raise ValueError(f"{k!r} in predictor: {v!r}")
    return Dataset(out)


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
    columns
        Only applies to NUTS (:class:`Dataset`) predictors.
        Use a single file with different columns. The code is interpreted as
        ``{name}-{value-column}-{mask-column}-{NUTS-method}`` (the last two are
        optional).

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
     - ``mask``: If present, the (boolean) mask will be applied to ``value``
       (``value`` will be set to zero wherever ``mask`` is ``False``).

    The variables supports the following randomization protocols:

     - ``$permute``: Shuffle the values. If ``mask`` is present in the dataset,
       only shuffle the cases for which ``mask == True``. An alternative mask,
       ``ds['mask_key']``, can be specified as ``$[mask_key]permute``.
     - ``$remask``: Shuffle ``mask``. ``$[mask_key]remask`` can be used to limit
       shuffling of ``mask`` to cases specified by ``ds['mask_key']``.
     - ``$shift``: displace the final uniform time-series circularly (i.e.,
       the impulse times themselves change).
    """
    def __init__(self, resample: str = None, columns: bool = False):
        assert resample in (None, 'bin', 'resample')
        self.resample = resample
        self.columns = columns

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
                raise IOError(f"{path.name} does not contain tstep={tstep!r}")
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

    def _generate(self, tmin: float, tstep: float, n_samples: int, code: Code, directory: Path):
        # predictor for one input file
        file_name = code.nuts_file_name(self.columns)
        x = self._load(tstep, file_name, directory)
        if isinstance(x, Dataset):
            if n_samples is None:
                raise ValueError(f"n_samples={n_samples!r}")
            uts = UTS(tmin, tstep, n_samples)
            x = self._ds_to_ndvar(x, uts, code)
        elif isinstance(x, NDVar):
            x = pad(x, tmin, nsamples=n_samples)
        else:
            raise RuntimeError(x)

        if code.shuffle in NDVAR_SHUFFLE_METHODS:
            x = shuffle(x, code.shuffle, code.shuffle_index, code.shuffle_angle)
            code.register_shuffle(index=True)
        return x

    def _generate_continuous(self, uts: UTS, ds: Dataset, stim_var: str, code: Code, directory: Path):
        # place multiple input files into a continuous predictor
        cache = {stim: self._load(uts.tstep, code.with_stim(stim).nuts_filename(self.columns), directory) for stim in ds[stim_var].cells}
        # determine type
        stim_type = {type(s) for s in cache.values()}
        assert len(stim_type) == 1
        stim_type = stim_type.pop()
        # generate x
        if stim_type is Dataset:
            dss = []
            for t, stim in ds.zip('T_relative', stim_var):
                x = cache[stim].copy()
                x['time'] += t
                dss.append(x)
                if code.nuts_method:
                    x_stop_ds = t_stop_ds(x, t)
                    dss.append(x_stop_ds)
            x = self._ds_to_ndvar(combine(dss), uts, code)
        elif stim_type is NDVar:
            v = cache[ds[0, stim_var]]
            dimnames = v.get_dimnames(first='time')
            dims = (uts, *v.get_dims(dimnames[1:]))
            shape = [len(dim) for dim in dims]
            x = NDVar(numpy.zeros(shape), dims, code.key)
            for t, stim in ds.zip('T_relative', stim_var):
                x_stim = cache[stim]
                i_start = uts._array_index(t + x_stim.time.tmin)
                i_stop = i_start + len(x_stim.time)
                x.x[i_start:i_stop] = x_stim.get_data(dimnames)
        else:
            raise RuntimeError(f"stim_type={stim_type!r}")
        return x

    def _ds_to_ndvar(self, ds: Dataset, uts: UTS, code: Code):
        if self.columns:
            column_key, mask_key = code.nuts_columns
            if column_key is None:
                column_key = 'value'
                ds[:, column_key] = 1
        else:
            column_key = 'value'
            mask_key = 'mask' if 'mask' in ds else None

        if mask_key:
            mask = ds[mask_key].x
            assert mask.dtype.kind == 'b', "'mask' must be boolean"
        else:
            mask = None

        if code.shuffle_index:
            shuffle_mask = ds[code.shuffle_index].x
            if shuffle_mask.dtype.kind != 'b':
                raise code.error("shuffle index must be boolean", -1)
            elif code.shuffle == 'permute' and mask is not None:
                assert not numpy.any(shuffle_mask[~mask])
        elif code.shuffle == 'permute':
            shuffle_mask = mask
        else:
            shuffle_mask = None

        if code.shuffle == 'remask':
            if mask is None:
                raise code.error("$remask for predictor without mask", -1)
            rng = code._get_rng()
            if shuffle_mask is None:
                rng.shuffle(mask)
            else:
                remask = mask[shuffle_mask]
                rng.shuffle(remask)
                mask[shuffle_mask] = remask
            code.register_shuffle(index=True)

        if mask is not None:
            ds[column_key] *= mask

        if code.shuffle == 'permute':
            rng = code._get_rng()
            if shuffle_mask is None:
                rng.shuffle(ds[column_key].x)
            else:
                values = ds[column_key].x[shuffle_mask]
                rng.shuffle(values)
                ds[column_key].x[shuffle_mask] = values
            code.register_shuffle(index=True)

        # prepare output NDVar
        if code.nuts_method == 'is':
            dim = Categorial('representation', ('step', 'impulse'))
            x = NDVar(numpy.zeros((2, len(uts))), (dim, uts), name=code.key)
            x_step, x_impulse = x
        else:
            x = NDVar(numpy.zeros(len(uts)), uts, name=code.key)
            if code.nuts_method == 'step':
                x_step, x_impulse = x, None
            elif not code.nuts_method:
                x_step, x_impulse = None, x
            else:
                raise code.error(f"NUTS-method={code.nuts_method!r}")

        # fill in values
        ds = ds[ds['time'] < uts.tstop]
        if x_impulse is not None:
            for t, v in ds.zip('time', column_key):
                x_impulse[t] = v
        if x_step is not None:
            t_stops = ds[1:, 'time']
            if ds[-1, column_key] != 0:
                if 'tstop' not in ds.info:
                    raise code.error("For step representation, the predictor datasets needs to contain ds.info['tstop'] to determine the end of the last step", -1)
                t_stops = chain(t_stops, [ds.info['tstop']])
            for t0, t1, v in zip(ds['time'], t_stops, ds[column_key]):
                x_step[t0:t1] = v
        return x


class MakePredictor:
    """Predictor calls .make_predictor"""
