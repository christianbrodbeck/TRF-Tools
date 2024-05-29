# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import chain
from pathlib import Path
from typing import List, Literal, Optional, Union

from eelbrain import load, Categorial, Dataset, Factor, NDVar, UTS, Var, combine, epoch_impulse_predictor, event_impulse_predictor, resample, set_time, set_tmin
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
    """Generate an impulse for each event

    For epoched data, one impulse per epoch;
    for continuous data, one impulse per event in the event list.

    Parameters
    ----------
    value
        Name of a :class:`Var` or :class:`Factor` in the events :class:`Dataset`
        (or expression resulting in one).
    latency
        Latency of the impulse relative to the event in seconds (or expression
        retrieving it from the events dataset).
    sel
        Subset of events.
    """
    def __init__(
            self,
            value: Union[float, str] = 1.,
            latency: Union[float, str] = 0.,
            sel: str = None,
    ):
        self.value = typed_arg(value, float, str)
        self.latency = typed_arg(latency, float, str)
        self.sel = typed_arg(sel, str)

    def _generate(self, uts: UTS, ds: Dataset, code: Code):
        assert code.stim is None
        if self.sel:
            raise NotImplementedError
        return epoch_impulse_predictor((ds.n_cases, uts), self.value, self.latency, code.code, ds)

    def _generate_continuous(self, uts: UTS, ds: Dataset, code: Code):
        assert code.stim is None
        if self.sel:
            ds = ds.sub(self.sel)
        return event_impulse_predictor(uts, 'T_relative', self.value, self.latency, code.code, ds)


class FilePredictorBase:

    def __init__(
            self,
            resample: Literal['bin', 'resample'] = None,
            sampling: Literal['continuous', 'discrete'] = None,
    ):
        assert resample in (None, 'bin', 'resample')
        self.resample = resample
        self.sampling = sampling

    def _resample(self, x: NDVar, tstep: float = None):
        if tstep is None or x.time.tstep == tstep:
            pass
        elif x.time.tstep > tstep:
            raise ValueError(f"Requested samplingrate rate is higher than in file ({1/tstep:g} > {1/x.time.tstep:g})")
        elif self.resample == 'bin':
            x = x.bin(tstep, label='start')
        elif self.resample == 'resample':
            srate = 1 / tstep
            int_srate = int(round(srate))
            srate = int_srate if abs(int_srate - srate) < .001 else srate
            x = resample(x, srate)
        elif self.resample is None:
            raise RuntimeError(f"{path.name} has tstep={x.time.tstep}, not {tstep}. Set the {self.__class__.__name__} resample parameter to enable automatic resampling.")
        else:
            raise RuntimeError(f"{self.resample=}")
        return x

    def _sampling(
            self,
            data_type: Literal['nuts', 'uts'] = None,
            nuts_method: str = None,
    ):
        if data_type == 'uts':
            return self.sampling or 'continuous'
        elif data_type == 'nuts' or nuts_method:
            if nuts_method == 'step':
                return 'continuous'
            elif nuts_method == 'is':
                return None
            elif nuts_method is None:
                return 'discrete'
            else:
                raise RuntimeError(f'{nuts_method=}')
        else:
            return self.sampling


class FilePredictor(FilePredictorBase):
    """Predictor stored in files corresponding to specific stimuli

    There are two basic ways to represent predictors in files (see the Notes
    section below for  details):

        1. Uniform time series (UTS). A :class:`NDVar` with time dimension
           matching the data.
        2. Non-uniform time series (NUTS). A :class:`Dataset` with columns
           representing time stamps, event values and optionally event masks.

    .. warning::
        When changing a file in which a predictor is stored, cached results
        using that predictor will not automatically be deleted. Use
        :meth:`TRFExperiment.invalidate` whenever replacing a predictors.

    Parameters
    ----------
    resample
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
        ``{name}-{value-column}-{mask-column}``. The code ``{name}`` alone
        invokes an intercept, i.e. a value of 1 at each time point.
    sampling
        Whether to expect a continuous or a discrete predictor (usually an
        :class:`NDVar` or a :class:`Dataset`, respectively). Used to decide
        whether to filter this predictor with ``filter_x='continuous'``.
        Note: ``discrete`` predictors with ``*-step`` suffix will always be
        trated as continuous (i.e. filtered).

    Notes
    -----
    The file-predictor expects to find a file for each stimulus containing the
    predictor at::

        {root}/predictors/{stimulus}~{name}[-{options}].pickle

    Where ``stimulus`` refers to the name provided by ``stim_var``, ``name``
    refers to the predictor's name, and the optional ``options`` can be used to
    distinguish different sub-varieties of the same predictor.

    UTS
    ^^^
    UTS predictors are stored as :class:`NDVar` objects with time dimension
    matching the data. The ``-{options}`` part of the filename can be used
    freely to store different predictors that are managed by the same
    :class:`FilePredictor` instance. Use the ``resample`` parameter to
    determine how the predictor is resampled to match the samplingrate of the
    data.

    NUTS
    ^^^^
    NUTS predictors are specified as :class:`Dataset` objects.
    When loading a predictor, :class:`Dataset`
    predictors are converted to uniform time series by placing impulses at
    time-stamps specified in the datasets.

    Without the ``columns`` option, the dataset is expected to contain the
    following columns:

     - ``time``: Time stamp of the event (impulse) in seconds.
     - ``value``: Value of the impulse (magnitude).
     - ``mask`` (optional): If present, the (boolean) mask will be applied to
       ``value`` (i.e., ``value`` will be set to zero wherever ``mask`` is
       ``False``).

    With the ``columns=True`` option, the columns containing the ``value`` and
    ``mask`` values can be specified dynamically in the variable name, as
    ``{name}-{value-column}`` or ``{name}-{value-column}-{mask-column}``.

    Supports the following randomization protocols:

     - ``$permute``: Shuffle the values. If ``mask`` is present in the dataset,
       only shuffle the cases for which ``mask == True``. An alternative mask,
       ``ds['mask_key']``, can be specified as ``$[mask_key]permute``.
     - ``$remask``: Shuffle ``mask``. ``$[mask_key]remask`` can be used to limit
       shuffling of ``mask`` to cases specified by ``ds['mask_key']``.
     - ``$shift``: displace the final uniform time-series circularly (i.e.,
       the impulse times themselves change).
    """
    def __init__(
            self,
            resample: Literal['bin', 'resample'] = None,
            columns: bool = False,
            sampling: Literal['continuous', 'discrete'] = None,
    ):
        self.columns = columns
        super().__init__(resample, sampling)

    def _load(self, tstep: float, filename: str, directory: Path) -> NDVar:
        path = directory / f'{filename}.pickle'
        x = load.unpickle(path)
        # allow for pre-computed resampled versions
        if isinstance(x, list):
            xs = x
            for x in xs:
                if x.time.tstep == tstep:
                    break
            else:
                raise IOError(f"{path.name} does not contain {tstep=}")
        elif isinstance(x, NDVar):
            x = self._resample(x, tstep)
        elif not isinstance(x, Dataset):
            raise TypeError(f'{x!r} at {path}')
        return x

    def _generate(self, tmin: float, tstep: float, n_samples: int, code: Code, directory: Path):
        # predictor for one input file
        file_name = code.nuts_file_name(self.columns)
        x = self._load(tstep, file_name, directory)
        if isinstance(x, Dataset):
            if tmin is None:
                tmin = 0
            if tstep is None:
                tstep = 0.001
            if n_samples is None:
                if 'tstop' in x.info:
                    tstop = x.info['tstop']
                else:
                    tstop = x[-1, 'time'] + 0.5
                n_samples = int((tstop - tmin) // tstep)
            uts = UTS(tmin, tstep, n_samples)
            x = self._ds_to_ndvar(x, uts, code)
            x.info['sampling'] = self._sampling('nuts', code.nuts_method)
        elif isinstance(x, NDVar):
            if code.nuts_method:
                raise code.error(f"Suffix {code.nuts_method} reserved for non-uniform time series predictors")
            x = pad(x, tmin, nsamples=n_samples, set_tmin=True)
            x.info['sampling'] = self._sampling('uts')
        else:
            raise RuntimeError(x)

        if code.shuffle in NDVAR_SHUFFLE_METHODS:
            x = shuffle(x, code.shuffle, code.shuffle_index, code.shuffle_angle)
            code.register_shuffle(index=True)
        return x

    def _generate_continuous(
            self,
            uts: UTS,  # time axis for the output
            ds: Dataset,  # events
            stim_var: str,
            code: Code,
            directory: Path,
    ):
        # place multiple input files into a continuous predictor
        cache = {stim: self._load(uts.tstep, code.with_stim(stim).nuts_file_name(self.columns), directory) for stim in ds[stim_var].cells}
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
            x = NDVar.zeros(dims, code.key)
            for t, stim in ds.zip('T_relative', stim_var):
                x_stim = cache[stim]
                i_start = uts._array_index(t + x_stim.time.tmin)
                i_stop = i_start + len(x_stim.time)
                if i_stop > len(uts):
                    raise ValueError(f"{code.string_without_rand} for {stim} is longer than the data")
                x.x[i_start:i_stop] = x_stim.get_data(dimnames)
        else:
            raise RuntimeError(f"{stim_type=}")
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
            x = NDVar.zeros((dim, uts), name=code.key)
            x_step, x_impulse = x
        else:
            x = NDVar.zeros(uts, name=code.key)
            if code.nuts_method == 'step':
                x_step, x_impulse = x, None
            elif not code.nuts_method:
                x_step, x_impulse = None, x
            else:
                raise code.error(f"NUTS-method={code.nuts_method!r}")

        # fill in values
        ds = ds[ds['time'] < uts.tmax + uts.tstep / 2]
        if x_impulse is not None:
            for t, v in ds.zip('time', column_key):
                x_impulse[t] += v
        if x_step is not None:
            t_stops = ds[1:, 'time']
            if ds[-1, column_key] != 0:
                if 'tstop' not in ds.info:
                    raise code.error("For step representation, the predictor datasets needs to contain ds.info['tstop'] to determine the end of the last step", -1)
                t_stops = chain(t_stops, [ds.info['tstop']])
            for t0, t1, v in zip(ds['time'], t_stops, ds[column_key]):
                x_step[t0:t1] = v
        return x


class SessionPredictor(FilePredictorBase):
    """Predictor stored in files corresponding to specific subjects

    In contrast to a :class:`FilePredictor`, which represents a specific
    stimulus, a :class:`SessionPredictor` represents a whole recording session
    for a specific subject.

    Filename should be ``{subject} {session}~{code}.pickle`` or, if the
    experiment includes multiple visits,
    ``{subject} {session} {visit}~{code}.pickle``.
    """
    def _load(self, tstep: Optional[float], filename: str, directory: Path) -> NDVar:
        path = directory / f'{filename}.pickle'
        x = load.unpickle(path)
        x = self._resample(x, tstep)
        return x

    def _generate(
            self,
            tmin: float,
            tstep: float,
            n_samples: int,
            code: Code,
            directory: Path,
            subject: str,
            recording: str,
    ):
        "predictor for one recording"
        if code.stim is not None:
            raise code.error(f"{self.__class__.__name__} cannot have stimulus", -1)
        elif code.nuts_method:
            raise code.error(f"Suffix {code.nuts_method} reserved for non-uniform time series predictors")
        elif code.shuffle:
            raise code.error(f"Shuffling not available for {self.__class__.__name__}")
        file_name = f"{subject} {recording}~{code.string}"
        x = self._load(tstep, file_name, directory)
        x = pad(x, tmin, nsamples=n_samples, set_tmin=True)
        x.info['sampling'] = self._sampling('uts')
        return x

    def _epoch_for_data(
            self,
            x: NDVar,
            utss: List[UTS],
            onset_times: List[float],  # onset of utss in x (relative to first uts)
    ) -> List[NDVar]:
        out = []
        for uts, t0 in zip(utss, onset_times):
            # align x to uts
            if t0:
                t0 = x.time.tstep * round(t0 / x.time.tstep)
                new_tmin = x.time.tmin - t0  # set x t=0 to uts t=0
                x_shifted = set_tmin(x, new_tmin)
            else:
                x_shifted = x
            # resample
            if x_shifted.time.tstep == uts.tstep:
                x_resampled = x_shifted
            else:
                x_cropped = pad(x_shifted, uts.tmin - 2, uts.tstop + 2)
                x_resampled = self._resample(x_cropped, uts.tstep)
            x_matching = pad(x_resampled, uts.tmin, uts.tstop, set_tmin=True)
            assert x_matching.time == uts
            out.append(x_matching)
        return out


class MakePredictor:
    """Predictor calls .make_predictor"""
