# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Eelbrain Experiment extension for analyzing continuous response models

=====
Terms
=====

Randomization
-------------

- Suffix demarcated by ``$`` for shuffling:  ``audspec8$shift``


Multiple streams
----------------

Prefix demarcated by |, indicating variable that names stimulus (defaulting
to ``e.stim_var``).


======
Models
======

Models are built out of terms, each term is specified by a code.
Shortcuts for models can be defined in :attr:`TRFExperiment.models`

Models can always be specified as combination of pre-defined models and terms,
joined with ``+``.


===========
Comparisons
===========

.. Note::
    Implementation in :mod:`trftools.pipeline._model`.

Examples assume ``x_model`` = ``x1 + x2 + ...`` etc. (in examples with only one
model ``model`` == ``x_model``).


Comparing models
^^^^^^^^^^^^^^^^

Whole models can be compared with comparisons indicating tailedness::

    x_model = y_model
    x_model > y_model

Furthermore, shortcuts exist for testing model components. These differ on
whether model comparison is based on cross-validation or shuffling. With
cross-validation (``cv=True``):

``x_model | x2``
    Test the contribution of ``x2`` to ``x_model``. Compare the complete
    ``x_model`` to ``x_model`` with ``x2`` removed. The right side can contain
    multiple components, e.g. ``x_model | x2 + x3``.
``x_model +| y``
    Test the effect of adding ``y`` to ``x_model``, i.e., equivalent to
    ``x_model + y > x_model``.
``x_model | y1 = y2``
    Equivalent tp ``x_model + y1 = x_model + y2``.

With predictor randomization (``cv=False``):

``x_model | x1$rand``
    Test the contribution of ``x2`` to ``x_model``. Compare the complete
    ``x_model`` to ``x_model`` with ``x2`` randomized.
``x_model +| y$rand``
    Test the effect of adding ``y`` to ``x_model``, compared to adding a
    randomized version of ``y``. Equivalent to``x_model + y > x_model + y$rand``.

"""
from collections import defaultdict
import fnmatch
from functools import partial
from glob import glob
from itertools import repeat
from operator import attrgetter
import os
from os.path import exists, getmtime, join, relpath, splitext
from pathlib import Path
from pyparsing import ParseException
import re
from typing import Union

import eelbrain
from eelbrain import (
    fmtxt, load, save, table, plot, testnd,
    MultiEffectNDTest, BoostingResult,
    MneExperiment, Dataset, Datalist, Factor, NDVar, Categorial, UTS,
    morph_source_space, rename_dim, boosting, combine, concatenate,
)
from eelbrain._exceptions import DimensionMismatchError
from eelbrain.fmtxt import List, Report, Table
from eelbrain.pipeline import TTestOneSample, TTestRelated, TwoStageTest, RawFilter, RawSource
from eelbrain._experiment.definitions import FieldCode
from eelbrain._experiment.epochs import EpochCollection
from eelbrain._experiment.mne_experiment import DataArg, PMinArg, DefinitionError, TestDims, guess_y, cache_valid
from eelbrain._data_obj import legal_dataset_key_re, isuv
from eelbrain._io.pickle import update_subjects_dir
from eelbrain._text import ms, n_of
from eelbrain._utils.mne_utils import is_fake_mri
from eelbrain._utils.numpy_utils import newaxis
from eelbrain._utils import ask
from filelock import FileLock
import numpy as np
from tqdm import tqdm

from .._ndvar import pad
from .._numpy_funcs import arctanh
from ._code import Code
from ._jobs import TRFsJob, ModelJob
from ._model import Comparison, Model, StructuredModel, load_models, model_comparison_table, model_name_parser, save_models
from ._predictor import EventPredictor, FilePredictor, MakePredictor
from ._results import ResultCollection
from . import _trf_report as trf_report


DATA_DEFAULT = 'source'
FIT_METRICS = ('r', 'z', 'r1', 'z1', 'residual', 'det')
UV_FUNCTIONS = ('sum', 'mean', 'max')
FIT_METRIC_RE = re.compile(rf"^({'|'.join(FIT_METRICS)})(?:\.({'|'.join(UV_FUNCTIONS)}))?$")
# https://en.wikipedia.org/wiki/Fisher_transformation

# templates for files affected by models; is_public
TRF_TEMPLATES = (
    ('trf-file', False),
    ('trf-test-file', False),
    ('model-test-file', False),
    ('model-report-file', True),
)
DSTRF_RE = re.compile(r'(ncrf)(?:-(\w+))?$')

ComparisonArg = Union[str, Comparison, StructuredModel]


class NameTooLong(Exception):

    def __init__(self, name):
        Exception.__init__(
            self,
            "Name too long (%i characters), consider adding a shortened model "
            "name: %s" % (len(name), name))


class FilenameTooLong(Exception):

    def __init__(self, path):
        Exception.__init__(
            self,
            "Filename too long (%i characters), consider adding a shortened "
            "model name: %s" % (len(os.path.basename(path)), path))


def split_model(x):
    return [v.strip() for v in x.split('+')]


def trf_test_parc_arg(y):
    if y.ndim == 4:
        for ydim in ['sensor', 'source']:
            if y.has_dim(ydim):
                break
        else:
            raise RuntimeError(f"{y} does not have sensor or source dimension")
        dim = y.get_dims((None, 'case', ydim, 'time'))[0]
        if isinstance(dim, Categorial):
            return dim.name


def difference_maps(dss):
    """Difference maps for model comparison"""
    diffs = {}
    subjects = None
    for x, x_ds in dss.items():
        ds = table.repmeas('z', 'model', 'subject', ds=x_ds)
        if subjects is None:
            subjects = ds['subject']
        else:
            assert np.all(ds['subject'] == subjects)
        diff = ds['test'] - ds['baseline']
        for hemi in ('lh', 'rh'):
            diffs[x, hemi] = diff.sub(source=hemi)
    return subjects, diffs


class TRFExperiment(MneExperiment):
    # Event variable that identifies stimulus files. To specify multiple
    # stimuli per event (e.g., foreground and background) use a dictionary
    # mapping {prefix: stim_var}, e.g. {'': 'fg', 'bg': 'bg', 'mix': 'mix'}
    # maps
    #  - 'audspec' -> stimulus from variable called 'fg'
    #  - 'bg|audspec' -> stimulus from variable called 'bg'
    stim_var = 'stimulus'
    predictors = {}

    # exclude rare phonemes
    # epoch -> list of phones
    exclude_phones = {}

    _values = {
        # Predictors
        'predictor-dir': join('{root}', 'predictors'),
        # TRF
        'trf-sdir': join('{cache-dir}', 'trf'),
        'trf-dir': join('{trf-sdir}', '{subject}'),
        'trf-file': join('{trf-dir}', '{analysis}', '{epoch_visit} {test_options}.pickle'),
        'trf-test-file': join('{cache-dir}', 'trf-test', '{analysis} {group}', '{folder}', '{test_desc}.pickle'),
        # model comparisons
        'model-test-file': join('{cache-dir}', 'model-test', '{analysis} {group}', '{folder}', '{test_desc}.pickle'),
        'model-res-dir': join('{root}', 'results-models'),
        'model-report-file': join('{model-res-dir}', '{analysis} {group}', '{folder}', '{test_desc}.html'),
        # predictors
        'predictor-cache-dir': join('{cache-dir}', 'predictors'),
    }

    models = {}
    _named_models = {}
    _model_names = {}
    _empty_test = True

    _parc_supersets = {}

    def _collect_invalid_files(self, invalid_cache, new_state, cache_state):
        rm = MneExperiment._collect_invalid_files(self, invalid_cache, new_state, cache_state)

        # stimuli
        for var, subject in invalid_cache['variable_for_subject']:
            if var in self._stim_var.values():
                state = {'subject': subject}
                rm['trf-file'].add(state)
                for group, members in self._groups.items():
                    if subject in members:
                        state = {'group': group}
                        rm['trf-test-file'].add(state)
                        rm['model-test-file'].add(state)
                        rm['model-report-file'].add(state)

        # epochs are based on events
        for subject, recording in invalid_cache['events']:
            for epoch, params in self._epochs.items():
                if recording not in params.sessions:
                    continue
                rm['trf-file'].add({'subject': subject, 'epoch': epoch})

        # group
        for group in invalid_cache['groups']:
            state = {'group': group}
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # raw
        for raw in invalid_cache['raw']:
            state = {'analysis': f'{raw} *'}
            rm['trf-file'].add(state)
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # epochs
        for epoch in invalid_cache['epochs']:
            state = {'epoch': epoch}
            rm['trf-file'].add(state)
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # cov
        for cov in invalid_cache['cov']:
            state = {'analysis': f'* {cov} *'}
            rm['trf-file'].add(state)
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # parcs
        for parc in invalid_cache['parc']:
            for opt in (f'{parc} *', f'* {parc} *'):
                state = {'test_options': opt}
                rm['trf-file'].add(state)
            state = {'folder': f'{parc} masked'}
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        return rm

    @classmethod
    def _eval_inv(cls, inv):
        if DSTRF_RE.match(inv):
            return inv
        else:
            return MneExperiment._eval_inv(inv)

    def _post_set_inv(self, _, inv):
        if DSTRF_RE.match(inv):
            inv = '*'
        MneExperiment._post_set_inv(self, _, inv)

    @staticmethod
    def _update_inv_cache(fields):
        if DSTRF_RE.match(fields['inv']):
            return fields['inv']
        return MneExperiment._update_inv_cache(fields)

    def _subclass_init(self):
        # predictors
        for key in self.predictors:
            if not legal_dataset_key_re.match(key):
                raise ValueError(f"{key!r}: invalid predictor key")
        # for model test
        self._field_values['test'] += ('',)
        # event variable that indicates stimuli
        if isinstance(self.stim_var, str):
            self._stim_var = {'': self.stim_var}
        elif isinstance(self.stim_var, dict):
            self._stim_var = self.stim_var.copy()
        else:
            raise TypeError(f"MneExperiment.stim_var={self.stim_var!r}")
        # structured models
        for name in self.models:
            try:
                model_name_parser.parseString(name, True)
            except ParseException:
                raise DefinitionError(f"{name!r}: invalid model name")
            if re.match(r'^[\w\-+|]*-red\d*$', name):
                raise ValueError(f"{name}: invalid model name (-red* pattern is reservered)")

        self._structured_models = {k: StructuredModel.coerce(v) for k, v in self.models.items()}
        self._structured_model_names = {m: k for k, m in self._structured_models.items()}
        # TODO: detect changes in structured models
        # load cache models:  {str: Model}
        self._model_names_file = join(self.get('cache-dir', mkdir=True), 'model-names.pickle')
        self._model_names_file_lock = FileLock(self._model_names_file + '.lock')
        with self._model_names_file_lock:
            self._load_model_names()

    def _load_model_names(self):
        if exists(self._model_names_file):
            self._named_models = load_models(self._model_names_file)
        else:
            self._named_models = {}
        self._model_names = {model.sorted_key: name for name, model in self._named_models.items()}

    def _register_model(self, model: Model) -> str:
        "Register a new model (generate a name)"
        model = model.without_randomization
        with self._model_names_file_lock:
            self._load_model_names()
            if model.sorted_key in self._model_names:
                return self._model_names[model.sorted_key]
            name = self._generate_model_name(model)
            self._named_models[name] = model
            save_models(self._named_models, self._model_names_file)
        self._model_names[model.sorted_key] = name
        return name

    def _generate_model_name(self, model: Model):
        assert not model.has_randomization
        if len(model.terms) == 1:
            if model.name not in self._named_models:
                return model.name
        for i in range(9999999):
            name = f"model{i}"
            if name not in self._named_models:
                return name
        raise RuntimeError("Ran out of model names...")

    def _find_model_files(self, name, trfs: bool = False, tests: bool = False):
        """Find all files associated with a model

        Will not find ``model (name$shift)``
        """
        assert trfs or tests
        if name not in self._named_models:
            raise ValueError(f"{name!r}: not a named model")
        files = []
        pattern = f"*{name}*"
        regex = re.compile(rf"(^| ){re.escape(name)}(-red\d+)?($|[. ])")  # -red for legacy names
        for temp, is_public in TRF_TEMPLATES:
            if temp.startswith('trf'):
                if not trfs:
                    continue
            if temp.startswith('model'):
                if not tests:
                    continue
            for path in self.glob(temp, True, test_options=pattern):
                if regex.search(path):
                    files.append(path)
        return files

    def _remove_model(self, name, files=None):
        """Remove a named model and delete all associated files

        .. warning::
            Only use this command when only a single instance of this
            TRFExperiment class exists.
            A separate, previously created instance of the same TRFExperiment
            class would retain a reference to this model name and might use it
            again, leading to conflicts later.
        """
        # FIXME: check model names file mtime any time model names are accessed?
        if files is None:
            files = self._find_model_files(name, trfs=True, tests=True)
        if files:
            while True:
                command = ask(f"Remove model {name} and delete {len(files)} files?", {"remove": "confirm", 'show': 'list files'}, allow_empty=True)
                if command == 'remove':
                    break
                elif command == 'show':
                    prefix = os.path.commonprefix(files)
                    n_prefix = len(prefix)
                    print(f"At {prefix}:")
                    print('\n'.join(f'  {path[n_prefix:]}' for path in files))
                else:
                    raise RuntimeError("Model deletion aborted")

        for path in files:
            os.remove(path)

        with self._model_names_file_lock:
            self._load_model_names()
            model = self._named_models.pop(name)
            del self._model_names[model.sorted_key]
            save_models(self._named_models, self._model_names_file)

    # Stimuli
    #########
    def add_predictors(self, ds, model, filter=False, y=None):
        """Add all predictor variables in a given model to a :class:`Dataset`

        Parameters
        ----------
        ds : Dataset
            Dataset with the dependent measure.
        model : str
            Model for which to load predictors.
        filter : bool | str
            Filter predictors. Name of a raw pipe, or ``True`` to use current
            raw setting; default ``False``).
        y : str
            :class:`NDVar` to match time axis to.
        """
        x = self._coerce_model(model)
        for term in x.terms:
            code = Code.coerce(term.string)  # TODO: use parse result
            self.add_predictor(ds, code, filter, y)

    def add_predictor(self, ds, code, filter=False, y=None):
        """Add predictor variable to a :class:`Dataset`

        Parameters
        ----------
        ds : Dataset
            Dataset with the dependent measure.
        code : str
            Predictor to add. Suffix demarcated by ``$`` for shuffling.
        filter : bool | str
            Filter predictors. Name of a raw pipe, or ``True`` to use current
            raw setting; default ``False``).
        y : str | NDVar | UTS
            :class:`NDVar` to match time axis to.
        """
        if isinstance(y, UTS):
            time = y
        else:
            if y is None:
                y = ds[guess_y(ds)]
            elif isinstance(y, str):
                y = ds[y]
            if isinstance(y, NDVar):
                time = y.time
            else:
                time = [yi.time for yi in y]
        is_variable_time = isinstance(time, list)
        code = Code.coerce(code)
        code.seed(ds.info['subject'])

        try:
            predictor = self.predictors[code.next()]
        except KeyError:
            raise code.error(f"predictor undefined in {self.__class__.__name__}", 0)

        # which Dataset variable indicates the stimulus?
        stim_var = self._stim_var[code.stim or '']

        # For nested events
        events_key = ds.info.get('nested_events')
        if events_key:
            assert is_variable_time
            directory = Path(self.get('predictor-dir'))
            xs = []
            assert not code._shuffle_done
            for uts, sub_ds in zip(time, ds[events_key]):
                code._shuffle_done = False  # each iteration will register shuffle
                if isinstance(predictor, EventPredictor):
                    x = predictor._generate_continuous(uts, sub_ds, code)
                elif isinstance(predictor, FilePredictor):
                    x = predictor._generate_continuous(uts, sub_ds, stim_var, code, directory)
                else:
                    raise RuntimeError(predictor)
                xs.append(x)
            ds[code.key] = xs
            return

        if isinstance(predictor, EventPredictor):
            assert not filter, f"filter not available for {predictor.__class__.__name__}"
            assert not is_variable_time, "EventPredictor not implemented for variable-time epoch"
            ds[code.key] = predictor._generate(time, ds, code)
            code.assert_done()
            return

        # load predictors (cache for same stimulus unless they are randomized)
        if code.has_randomization or is_variable_time:
            time_dims = time if is_variable_time else repeat(time, ds.n_cases)
            xs = [self.load_predictor(code.with_stim(stim), time.tstep, time.nsamples, time.tmin, filter, code.key) for stim, time in zip(ds[stim_var], time_dims)]
        else:
            x_cache = {stim: self.load_predictor(code.with_stim(stim), time.tstep, time.nsamples, time.tmin, filter, code.key) for stim in ds[stim_var].cells}
            xs = [x_cache[stim] for stim in ds[stim_var]]

        if not is_variable_time:
            xs = combine(xs)
        ds[code.key] = xs

    def prune_models(self):
        """Remove internal models that have no corresponding files

        See Also
        --------
        .remove_model
        .show_models
        """
        models = list(self._named_models)
        for model in models:
            for _ in self._find_model_files(model, trfs=True, tests=True):
                break
            else:
                self._remove_model(model, files=[])

    def model_job(self, x, report=True, reduce_model=False, **kwargs):
        """Compute all TRFs needed for a model-test

        Parameters
        ----------
        x : str
            Model or comparison.
        report : bool
            Schedule a model-test report.
        reduce_model : bool
            Iteratively reduce the model until it only contains predictors
            significant at the .05 level.
        priority : bool
            Prioritize job over others (default ``False``)
        ...
            For more arguments see :meth:`.load_model_test`.

        See Also
        --------
        .trf_job
        """
        return ModelJob(x, self, report, reduce_model, **kwargs)

    def trf_job(self, x, **kwargs):
        """Compute all TRFs with the given model

        Parameters
        ----------
        x : str
            Model.
        priority : bool
            Prioritize job over others (default ``False``)
        ...
            For more arguments see :meth:`.load_trf`.

        See Also
        --------
        .model_job
        """
        return TRFsJob(x, self, **kwargs)

    def load_predictor(self, code, tstep=0.01, n_samples=None, tmin=0., filter=False, name=None):
        """Load predictor NDVar

        Parameters
        ----------
        code : str
            Code for the predictor to load (using the pattern
            ``{stimulus}|{code}${randomization}``)
        tstep : float
            Time step for the predictor.
        n_samples : int
            Number of samples in the predictor (the default returns all
            available samples).
        tmin : scalar
            First sample time stamp (default 0).
        filter : bool
            Filter the predictor with the same method as the raw data.
        name : str
            Reassign the name of the predictor :class:`NDVar`.
        """
        code = Code.coerce(code)
        try:
            predictor = self.predictors[code.lookahead()]
        except KeyError:
            raise code.error(f"predictor undefined in {self.__class__.__name__}", 0)

        # if called without add_predictor
        if code._seed is None:
            code.seed(self.get('subject'))

        if isinstance(predictor, FilePredictor):
            directory = Path(self.get('predictor-dir'))
            x = predictor._generate(tmin, tstep, n_samples, code, directory)
            code.register_string_done()
            code.assert_done()
        elif isinstance(predictor, MakePredictor):
            x = self._make_predictor(code, tstep, n_samples, tmin)
        elif isinstance(predictor, EventPredictor):
            raise ValueError(f"{code.string!r}: can't load {predictor.__class__.__name__} without data; use {self.__class__.__name__}.add_predictor()")
        else:
            raise code.error(f"Unknown predictor type {predictor}", 0)

        if filter:
            if filter is True:
                raw = self.get('raw')
            elif filter in self._raw:
                raw = filter
            else:
                raise ValueError(f"filter={filter!r}")
            pipe = self._raw[raw]
            pipes = []
            while not isinstance(pipe, RawSource):
                if isinstance(pipe, RawFilter):
                    pipes.append(pipe)
                pipe = pipe.source
            for pipe in reversed(pipes):
                x = pipe.filter_ndvar(x)

        if name is not None:
            x.name = name
        return x

    def load_predictors(self, stim, model, tstep=0.01, n_samples=None, tmin=0.):
        "Multiple predictors corresponding to ``model`` in a list"
        model = self._coerce_model(model)
        out = []
        for term in model.terms:
            y = self.load_predictor(f'{stim}|{term.string}', tstep, n_samples, tmin)
            out.append(y)
        return out

    def _make_predictor(self, code, tstep=0.01, n_samples=None, tmin=0., seed=False):
        "Wrapper for .make_predictor() with caching"
        if code.has_randomization:
            cache_path = None
        else:
            cache_dir = Path(self.get('predictor-cache-dir', mkdir=True))
            cache_path = cache_dir / f'{code.string} {tmin:g} {tstep:g}.pickle'
        # load/generate predictor
        if cache_path and exists(cache_path):
            x = load.unpickle(cache_path)
        else:
            x = self.make_predictor(code, tstep, n_samples, tmin, seed)
            code.assert_done()
        # cache
        if cache_path:
            save.pickle(x, cache_path)
        # match time axis
        x = pad(x, tmin, nsamples=n_samples)
        # check time
        if n_samples is None:
            target_time = UTS(tmin, tstep, x.time.nsamples)
        else:
            target_time = UTS(tmin, tstep, n_samples)
        if x.time != target_time:
            raise code.error(f"Predictor time {x.time} does not match requested time {target_time}")
        return x

    def make_predictor(self, code, tstep=0.01, n_samples=None, tmin=0., seed=False):
        raise NotImplementedError

    # TRF
    #####
    def load_trf(self, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, backward=False, make=False, path_only=False, **state):
        """TRF estimated with boosting

        Parameters
        ----------
        x : Model
            One or more predictor variables, joined with '+'.
        tstart : scalar
            Start of the TRF in s (default 0).
        tstop : scalar
            Stop of the TRF in s (default 0.5).
        basis : scalar
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions : int
            Number of partitions used for cross-validation in boosting (default
            is the number of epochs; -1 to concatenate data).
        samplingrate : int
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask : str
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta : scalar
            Boosting delta.
        mindelta : scalar < delta
            Boosting parameter.
        filter_x : bool
            Filter ``x`` with the last filter of the pipeline for ``y``.
        selective_stopping : int
            Stop boosting each predictor separately.
        cv : bool
            Cross-validation.
        data : 'sensor' | 'source'
            Data which to use.
        backward : bool
            Backward model (default is forward model).
        make : bool
            If the TRF does not exists, make it (the default is to raise an
            IOError).
        path_only : bool
            Return the path instead of loading the TRF.

        Returns
        -------
        res : BoostingResult
            Estimated model.
        """
        data = TestDims.coerce(data)
        x = self._coerce_model(x)
        # check epoch
        epoch = self._epochs[self.get('epoch')]
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r} (use .load_trfs() to load multiple TRFs from a collection epoch)")
        # check cache
        dst = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, **state)
        if path_only:
            return dst
        elif exists(dst) and cache_valid(getmtime(dst), self._epochs_mtime()):
            try:
                res = load.unpickle(dst)
            except:
                print(f"Error unpickling {dst}")
                raise
            if data.source:
                update_subjects_dir(res, self.get('mri-sdir'), 2)
            # TRFs from before storing n_samples
            if isinstance(res, BoostingResult) and res.n_samples is None:
                self._log.info("Recovering missing n_samples...")
                meg = self.load_epochs(samplingrate=samplingrate)['meg']
                res.n_samples = meg.shape[0] * meg.shape[meg.get_axis('time')]
                save.pickle(res, dst)
            # check x
            if not backward and hasattr(res, 'x'):  # not NCRF
                # res.x are from a Dataset (except variable length epochs)
                res_keys = [res.x] if isinstance(res.x, str) else res.x
                res_keys = sorted(Dataset.as_key(x) for x in res_keys)
                x_keys = sorted(Dataset.as_key(term) for term in x.term_names)
                if res_keys != x_keys:
                    res_model = Model.from_string(res_keys)
                    x_model = Model.from_string(x_keys)
                    table = model_comparison_table(x_model, res_model, 'x', 'cached result')
                    res_key = '+'.join(res_keys)
                    for name, model in self._named_models.items():
                        if res_key == model.dataset_based_key:
                            table.caption(f"Looks like {name}")
                            break
                    else:
                        table.caption("Model not recognized")
                    raise RuntimeError(f"Result x mismatch:\n{dst}\n{table}")
            return res

        # try to load from superset parcellation
        if not data.source:
            pass
        elif not mask:
            assert self.get('src')[:3] == 'vol'
        elif mask in self._parc_supersets:
            for super_parc in self._parc_supersets[mask]:
                try:
                    res = self.load_trf(x, tstart, tstop, basis, error, partitions, samplingrate, super_parc, delta, mindelta, filter_x, selective_stopping, cv, data, backward)
                except IOError:
                    pass
                else:
                    # make sure parc exists
                    self.make_annot(parc=mask)
                    res._set_parc(mask)
                    return res

        # make it if make=True
        if not make:
            raise IOError(f"TRF {relpath(dst, self.get('root'))} does not exist; set make=True to compute it.")

        self._log.info("Computing TRF:  %s %s %s %s", self.get('subject'), data.string, '->' if backward else '<-', x.name)
        func = self._trf_job(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward)
        if func is None:
            return load.unpickle(dst)
        res = func()
        save.pickle(res, dst)
        return res

    def _locate_trf(self, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, backward=False, **state):
        "Return path of the corresponding trf-file"
        self._set_trf_options(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, state=state)

        path = self.get('trf-file', mkdir=True)
        if len(os.path.basename(path)) > 255:
            raise FilenameTooLong(path)
        elif exists(path) and not cache_valid(getmtime(path), self._epochs_mtime()):
            os.remove(path)
        return path

    def _trf_job(self, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, backward=False, **state):
        "Return ``func`` to create TRF result"
        data = TestDims.coerce(data)
        epoch = self.get('epoch', **state)
        assert not isinstance(self._epochs[epoch], EpochCollection)
        x = self._coerce_model(x)
        if not x:
            return

        if data.source:
            inv = self.get('inv')
            m = DSTRF_RE.match(inv)
            if m:
                data = TestDims('sensor')
            else:
                morph = is_fake_mri(self.get('mri-dir'))
                data = TestDims.coerce(data, morph=morph)
        else:
            inv = m = None

        # NCRF: Cross-validations
        ncrf_args = {'mu': 'auto'}
        if m and m.group(2):
            # maybe model exists
            ncrf_tag = m.group(2)
            # find best mu from previous cross-validations
            with self._temporary_state:
                cv = self.load_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv=m.group(1))

            if ncrf_tag == 'l2':
                ncrf_args['mu'] = cv.cv_mu('l2')
            elif ncrf_tag == 'l2mu':
                ncrf_args['mu'] = cv.cv_mu('l2/mu')
            elif ncrf_tag == 'cv2':
                grade = 10
                cv_results = sorted(cv._cv_results, key=attrgetter('mu'))
                best_cv = min(cv_results, key=attrgetter('cross_fit'))
                i = cv_results.index(best_cv)
                ncrf_args['mu'] = np.logspace(np.log10(cv_results[i-1].mu), np.log10(cv_results[i+1].mu), grade+2)[1:-1]
            elif ncrf_tag == '50it':
                ncrf_args['n_iter'] = 50
            elif ncrf_tag == 'no_champ':
                ncrf_args.update(n_iter=1, n_iterf=1000, n_iterc=0)
            else:
                raise RuntimeError(f'inv={inv!r}')
            # check whether fit with mu exists
            if ncrf_tag.startswith('l2'):
                src_inv = None
                if ncrf_args['mu'] == cv.mu:
                    src_inv = 'dstrf'
                elif inv == 'dstrf-l2mu':
                    with self._temporary_state:
                        l2_trf = self.load_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv='dstrf-l2')
                    if ncrf_args['mu'] == l2_trf.mu:
                        src_inv = 'dstrf-l2'
                # if fit with mu exists, link it
                if src_inv is not None:
                    with self._temporary_state:
                        src = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv=src_inv)
                        dst = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv=inv)
                    os.link(src, dst)
                    return

        # load data
        if m:
            ds = self.load_epochs(samplingrate=samplingrate, data=data)
        elif data.source is True:
            # FIXME: in TRF-path, use mri-subject rather than mri value
            ds = self.load_epochs_stc(baseline=False, mask=mask, samplingrate=samplingrate, morph=data.morph)
        elif data.sensor:
            ds = self.load_epochs(samplingrate=samplingrate, data=data, interpolate_bads=data.sensor is True)
        else:
            raise NotImplemented(f"data={data.string!r}")
        y = ds[data.y_name]
        is_variable_time = isinstance(y, Datalist)
        # load predictors
        xs = []
        for term in sorted(x.term_names):
            code = Code.coerce(term)
            self.add_predictor(ds, code, filter_x, data.y_name)
            xs.append(ds[code.key])

        # determine partitions for NCRF
        if m:
            assert partitions is None
            if is_variable_time:
                partitions = 1
            elif (y.time.nsamples * y.time.tstep) / tstop < 30:
                # make sure chunk size is at least 30 TRFs
                partitions = -1
            else:
                partitions = 1

        # reshape data
        if partitions < 0:
            partitions = None if partitions == -1 else -partitions
            y = concatenate(y)
            xs = [concatenate(x) for x in xs]
        elif partitions is None:
            if not 3 <= ds.n_cases <= 10:
                raise TypeError(f"partitions=None: can't infer partitions parameter for {ds.n_cases} cases")

        if len(xs) == 1:
            xs = xs[0]
            if backward:
                y, xs = xs, y
        elif backward:
            raise ValueError("backward model with more than one stimulus")
        else:
            # check that all x have unique names
            names = [x_.name for x_ in xs]
            if len(set(names)) < len(names):
                raise ValueError(f"Multiple predictors with same name: {', '.join(names)}")
            if m and is_variable_time:
                # [[u1, u2], [v1, v2]] -> [[u1, v1], [u2, v2]]
                xs = list(zip(*xs))

        if m:
            y0 = y[0] if is_variable_time else y
            fwd = self.load_fwd(ndvar=True)
            cov = self.load_cov()
            chs = sorted(set(cov.ch_names).intersection(y0.sensor.names))
            if len(chs) < len(y0.sensor):
                if is_variable_time:
                    y = [yi.sub(sensor=chs) for yi in y]
                else:
                    y = y.sub(sensor=chs)
            else:
                assert np.all(y0.sensor.names == chs)
            from ncrf import fit_ncrf
            return partial(fit_ncrf, y, xs, fwd, cov, tstart, tstop, normalize=True, in_place=True, **ncrf_args)
        return partial(boosting, y, xs, tstart, tstop, 'inplace', delta, mindelta, error, basis, partitions=partitions, test=cv, selective_stopping=selective_stopping)

    def load_trfs(self, subject, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, backward=False, make=False, scale=None, smooth=None, smooth_time=None, vardef=None, permutations=1, vector_as_norm=False, **state):
        """Load TRFs for the group in a Dataset (see ``.load_trf()``)

        Parameters
        ----------
        subject : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group.
        x : Model
            One or more predictor variables, joined with '+'.
        tstart : scalar
            Start of the TRF in s (default 0).
        tstop : scalar
            Stop of the TRF in s (default 0.5).
        basis : scalar
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions : int
            Number of partitions used for cross-validation in boosting (default
            is the number of epochs; -1 to concatenate data).
        samplingrate : int
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask : str
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta : scalar
            Boosting delta.
        mindelta : scalar < delta
            Boosting parameter.
        filter_x : bool
            Filter ``x`` with the last filter of the pipeline for ``y``.
        selective_stopping : int
            Stop boosting each predictor separately.
        data : 'sensor' | 'source'
            Data which to use.
        backward : bool
            Backward model (default is forward model).
        make : bool
            If a TRF does not exists, make it (the default is to raise an
            IOError).
        scale : 'original'
            Rescale TRFs to the scale of the source data (default is the scale
            based on normalized predictors and responses).
        smooth : float
            Smooth TRFs (spatial smoothing, in [m] STD of Gaussian).
        smooth_time : str | float
            Smooth TRFs temporally.
        vardef : str
            Add variables for a given test.
        permutations : int
            When loading a partially permuted model, average the result
            of ``permutations`` different permutations.
        vector_as_norm : bool
            For vector data, return the norm at each time point instead of the
            vector.
        ...
            Experiment state parameters.

        Returns
        -------
        trf_ds : Dataset
            Dataset with the following variables: ``subject``, ``r``
            (correlation map) and one NDVar for each component of the TRF.
            ``trf_ds.info['xs']`` is a tuple of the names of all TRF components.
        """
        data = TestDims.coerce(data)
        subject, group = self._process_subject_arg(subject, state)
        x = self._coerce_model(x)

        # group data
        if group is not None:
            dss = []
            for _ in self.iter(group=group, progress_bar="Load TRFs"):
                ds = self.load_trfs(1, x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, scale, smooth, smooth_time, vardef, permutations, vector_as_norm)
                dss.append(ds)

            try:
                out = combine(dss)
            except DimensionMismatchError:
                if not backward:
                    raise
                # backward model can have incompatible sensor dimensions
                for ds in dss:
                    del ds[data.y_name]
                out = combine(dss)
                out.info['load_trfs'] = f"Dropping {data.y_name} (incompatible dimension)"
            return out

        # collection epochs
        epoch = self._epochs[self.get('epoch')]
        if isinstance(epoch, EpochCollection):
            dss = []
            with self._temporary_state:
                for sub_epoch in epoch.collect:
                    ds = self.load_trfs(1, x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, scale, smooth, smooth_time, None, permutations, vector_as_norm, epoch=sub_epoch)
                    ds[:, 'epoch'] = sub_epoch
                    dss.append(ds)
            ds = combine(dss)
            self._add_vars(ds, vardef, groupvars=True)
            return ds

        # single subject and epoch
        if permutations == 1:
            xs = (x,)
        elif not x.has_randomization:
            raise ValueError(f"permutations={permutations!r} for model without randomization ({x.name})")
        else:
            xs = x.multiple_permutations(permutations)

        if data.source:
            inv = self.get('inv')
            is_dstrf = bool(DSTRF_RE.match(inv))
            is_vector_data = is_dstrf or inv.startswith('vec')
        else:
            is_vector_data = is_dstrf = False

        # load result(s)
        h = r = z = r1 = z1 = residual = det = tstep = x_keys = res_partitions = None
        for x_ in xs:
            res = self.load_trf(x_, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make)
            # kernel
            if scale is None:
                res_h = res.h
            elif scale == 'original':
                res_h = res.h_scaled
            else:
                raise ValueError(f"scale={scale!r}")
            # make sure h is a tuple
            if isinstance(res_h, NDVar):
                res_h = [res_h]
            # morph to average brain
            if is_dstrf:
                res_h = [morph_source_space(h, 'fsaverage') for h in res_h]
            # for vector results, the average norm is relevant
            if is_vector_data and vector_as_norm:
                res_h = [hi.norm('space') for hi in res_h]
            # combine results
            if h is None:
                h = res_h
                if is_dstrf:
                    tstep = res.tstep
                else:
                    res_partitions = res.partitions
                    r = res.r
                    z = arctanh(r, info={'unit': 'z(r)'})
                    if is_vector_data:
                        r1 = res.r_l1
                        z1 = arctanh(r1, info={'unit': 'z(r)'})
                    residual = res.residual
                    det = res.proportion_explained
                    tstep = res.h_time.tstep
                x_keys = [Dataset.as_key(term) for term in x_.term_names]
            else:
                for hi, res_hi in zip(h, res_h):
                    hi += res_hi
                if not is_dstrf:
                    assert res.partitions == res_partitions
                    r += res.r
                    z += arctanh(res.r)
                    if r1 is not None:
                        r1 += res.r_l1
                        z1 += arctanh(res.r_l1)
                    residual += res.residual
                    det += res.proportion_explained
        # average
        if permutations > 1:
            for hi in h:
                hi /= permutations
            if not is_dstrf:
                r /= permutations
                z /= permutations
                if is_vector_data:
                    r1 /= permutations
                    z1 /= permutations
                residual /= permutations
                det /= permutations

        # output Dataset
        ds = Dataset(info={'xs': x_keys, 'x_names': x.term_names, 'samplingrate': 1 / tstep, 'partitions': partitions or res_partitions}, name=self._x_desc(x))
        ds['subject'] = Factor([subject], random=True)
        if not is_dstrf:
            ds[:, 'r'] = r
            ds[:, 'residual'] = residual
            ds[:, 'det'] = det
            ds[:, 'z'] = z
            if is_vector_data:
                ds['r1'] = r1[newaxis]
                ds['z1'] = z1[newaxis]

        # add kernel to dataset
        for hi in h:
            if smooth:
                if data.source is True:
                    hi = hi.smooth('source', smooth, 'gaussian')
                else:
                    raise ValueError(f"smooth={smooth!r} with data={data.string!r}")
            if smooth_time:
                hi = hi.smooth('time', smooth_time)
            ds[Dataset.as_key(hi.name)] = hi[newaxis]

        self._add_vars(ds, vardef, groupvars=True)
        return ds

    def _locate_missing_trfs(self, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, backward=False, permutations=1, existing=False, **state):
        "Return ``(path, state, args)`` for ._trf_job() for each missing trf-file"
        data = TestDims.coerce(data)
        x = self._coerce_model(x)
        if not x:
            return ()
        if state:
            self.set(**state)

        out = []
        args = (x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward)

        # multiple permutations
        if permutations > 1 and x.has_randomization:
            args = args[1:]
            for xi in x.multiple_permutations(permutations):
                out.extend(self._locate_missing_trfs(xi, *args, existing=existing))
            return out

        # EpochCollection: separate TRF for member epochs
        epoch = self._epochs[self.get('epoch')]
        if isinstance(epoch, EpochCollection):
            with self._temporary_state:
                for epoch_ in epoch.collect:
                    out.extend(self._locate_missing_trfs(*args, existing=existing, epoch=epoch_))
            return out

        # one model, one epoch
        for _ in self:
            path = self._locate_trf(*args)
            if not existing:
                if os.path.exists(path):
                    continue  # TRF exists for requested mask
                # Check whether TRF exists for superset parc
                super_exists = False
                for super_parc in self._parc_supersets.get(mask, ()):
                    spath = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, super_parc, delta, mindelta, filter_x, selective_stopping, cv, data, backward)
                    if os.path.exists(spath):
                        super_exists = True
                        break
                if super_exists:
                    continue
            out.append((path, self._copy_state(), args))
        return out

    def _xhemi_parc(self):
        parc = self.get('parc')
        with self._temporary_state:
            if parc == 'lobes':
                parc = 'cortex'
                self.set(parc=parc)
            self.make_annot(mrisubject='fsaverage_sym')
            self.make_src()
        return parc

    def load_trf_test(self, x, tstart=0, tstop=0.5, basis=0.05, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, permutations=1, make=False, make_trfs=False, scale=None, smooth=None, smooth_time=None, pmin='tfce', samples=10000, test=True, return_data=False, xhemi=False, xhemi_smooth=0.005, **state):
        """Load TRF test result

        Parameters
        ----------
        x : str
            One or more predictor variables, joined with '+'.
        tstart : scalar
            Start of the TRF in s (default 0).
        tstop : scalar
            Stop of the TRF in s (default 0.5).
        basis : scalar
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions : int
            Number of partitions used for cross-validation in boosting (default
            is the number of epochs; -1 to concatenate data).
        samplingrate : int
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask : str
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta : scalar
            Boosting delta.
        mindelta : scalar < delta
            Boosting parameter.
        filter_x : bool
            Filter ``x`` with the last filter of the pipeline for ``y``.
        selective_stopping : int
            Stop boosting each predictor separately.
        data : 'sensor' | 'source'
            Data which to use.
        make : bool
            If the test does not exists, make it (the default is to raise an
            IOError).
        make_trfs : bool
            If a TRF does not exists, make it (the default is to raise an
            IOError).
        scale : 'original'
            Rescale TRFs to the scale of the source data (default is the scale
            based on normalized predictors and responses).
        smooth : float
            Smooth data in space before test (value in [m] STD of Gaussian).
        smooth_time : str | float
            Smooth TRFs temporally.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 10,000).
        test : True | str
            Test to perform (default ``True`` is test against 0).
        return_data : bool
            Return the data along with the test result (see below).
        xhemi : bool
            Test between hemispheres.
        xhemi_smooth : float
            Smooth TRFs before morphing to the other hemisphere (Gaussian std
            in [m]; default 0.005 (5 mm)).
        ...
            Experiment state parameters.

        Returns
        -------
        ds : Dataset (if return_data==True)
            Data that forms the basis of the test.
        res : ResultCollection
            Test results for the specified test.
        """
        data = TestDims.coerce(data)
        if state:
            self.set(**state)
        if data.source:
            inv = self.get('inv')
            is_vector_data = inv.startswith('vec')
        else:
            is_vector_data = False
        # determine whether baseline model is needed:
        if test is True and is_vector_data and not xhemi:
            assert not cv
            compare_with_baseline_model = True
        else:
            assert permutations == 1
            compare_with_baseline_model = False
        # vector data can not be tested against 0
        if compare_with_baseline_model:
            comparison = self._coerce_comparison(x, cv)
            if isinstance(comparison, StructuredModel):
                if return_data:
                    raise NotImplementedError("return_data=True for multiple comparisons")
                return ResultCollection({
                    cmp.test_term_name: self.load_trf_test(cmp, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, permutations, make, make_trfs, scale, smooth, smooth_time, pmin, samples, test, return_data, xhemi, xhemi_smooth) for cmp in comparison.comparisons(cv)
                })
            if comparison.baseline_term_name is None:
                raise ValueError(f"x={x!r}: no unique baseline term")
            y_key = Dataset.as_key(comparison.test_term_name)
            y_keys = None
        else:
            comparison = self._coerce_model(x)
            y_key = None
            y_keys = [Dataset.as_key(term) for term in comparison.term_names]

        if xhemi:
            if xhemi_smooth % 0.001:
                raise ValueError(f'xhemi_smooth={xhemi_smooth!r}; parameter in [m] needs to be integer number of [mm]')
            test_options = f'xhemi-abs-{int(xhemi_smooth * 1000)}'
        else:
            test_options = None
        self._set_trf_options(comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, pmin=pmin, is_group_result=True, scale=scale, smooth_source=smooth, smooth_time=smooth_time, test=test, test_options=test_options, permutations=permutations)

        # check if cached
        dst = self.get('trf-test-file', mkdir=True)
        if self._result_file_mtime(dst, data):
            res = load.unpickle(dst)
            if compare_with_baseline_model:
                res0 = res
            else:
                res0 = res[y_keys[0]]

            if res0.samples >= samples:
                if data.source:
                    update_subjects_dir(res, self.get('mri-sdir'), 2)
            elif not make:
                raise IOError(f"Test has {res[x].samples} samples, {samples} samples requested; set make=True to make with {samples} samples.")
            else:
                res = {}
        elif not make:
            raise IOError(f"TRF-test {relpath(dst, self.get('root'))} does not exist; set make=True to compute it.")
        else:
            res = {}
        res_modified = not res
        test_kwargs = self._test_kwargs(samples, pmin, None, None, data, None) if res_modified else None

        if compare_with_baseline_model:
            # returns single test
            if return_data or res_modified:
                ds = self.load_trfs(-1, comparison.x1, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make_trfs, scale=scale, smooth=smooth, smooth_time=smooth_time, vector_as_norm=True)
                ds0 = self.load_trfs(-1, comparison.x0, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make_trfs, scale=scale, smooth=smooth, smooth_time=smooth_time, permutations=permutations, vector_as_norm=True)
                y0_key = Dataset.as_key(comparison.baseline_term_name)
                assert np.all(ds['subject'] == ds0['subject'])
                if res_modified:
                    y = ds[y_key]
                    y0 = ds0[y0_key]
                    res = testnd.ttest_rel(y, y0, tail=1, **test_kwargs)
                    save.pickle(res, dst)
                if return_data:
                    ds = ds['subject', y_key]
                    ds[y0_key] = ds0[y0_key]
                    return ds, res
            return res
        elif res_modified or return_data:
            if xhemi:
                parc = self._xhemi_parc()
                trf_ds, trf_res = self.load_trf_test(comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, permutations, make, make_trfs, scale, smooth, smooth_time, pmin, test=test, return_data=True)

                test_obj = TTestRelated('hemi', 'lh', 'rh') if test is True else self.tests[test]
                # xhemi data
                if test is True:
                    ds = Dataset(info=trf_ds.info)
                    ds['subject'] = trf_ds['subject'].tile(2)
                    ds['hemi'] = Factor(('lh', 'rh'), repeat=trf_ds.n_cases)
                else:
                    ds = trf_ds

                for x in tqdm(y_keys, f"X-Hemi TRF-Tests for {comparison.name}"):
                    y = trf_ds[x].abs()
                    if xhemi_smooth:
                        y = y.smooth('source', xhemi_smooth, 'gaussian')
                    lh, rh = eelbrain.xhemi(y, parc=parc)
                    y = combine((lh, rh)) if test is True else lh - rh
                    # mask
                    mask_lh, mask_rh = eelbrain.xhemi(trf_res[x].p <= 0.05, parc=parc)
                    np.maximum(mask_lh.x, mask_rh.x, mask_lh.x)
                    mask = mask_lh > .5
                    y *= mask
                    # test
                    if res_modified:
                        test_kwargs['parc'] = trf_test_parc_arg(y)
                        res[x] = self._make_test(y, ds, test_obj, test_kwargs)
                    if return_data:
                        ds[x] = y
            else:
                test_obj = TTestOneSample() if test is True else self.tests[test]
                if isinstance(test_obj, TwoStageTest):
                    assert test_obj.model is None, "not implemented"
                    # stage 1
                    lms = {y: [] for y in y_keys}
                    dss = []
                    for subject in self:
                        ds = self.load_trfs(1, comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make_trfs, scale=scale, smooth=smooth, smooth_time=smooth_time, vardef=test_obj.vars, permutations=permutations)
                        if res_modified:
                            for y in y_keys:
                                lms[y].append(test_obj.make_stage_1(y, ds, subject))
                        if return_data:
                            dss.append(ds)
                    # stage 2
                    if res_modified:
                        for y in y_keys:
                            test_kwargs['parc'] = trf_test_parc_arg(ds[y])
                            res[y] = test_obj.make_stage_2(lms[y], test_kwargs)
                    # data to return
                    if return_data:
                        ds = combine(dss)
                else:
                    ds = self.load_trfs(-1, comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make_trfs, scale=scale, smooth=smooth, smooth_time=smooth_time, vardef=test_obj.vars, permutations=permutations, vector_as_norm=True)
                    if res_modified:
                        for x in tqdm(y_keys, f"TRF-Tests for {comparison.name}"):
                            test_kwargs['parc'] = trf_test_parc_arg(ds[x])
                            res[x] = self._make_test(x, ds, test_obj, test_kwargs)

            if res_modified:
                save.pickle(res, dst)
        else:
            ds = None

        res = ResultCollection({term.string: res[Dataset.as_key(term.string)] for term in comparison.terms})

        if return_data:
            return ds, res
        else:
            return res

    def _set_trf_options(self, x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward=False, pmin=None, is_group_result=False, metric=None, scale=None, smooth_source=None, smooth_time=None, is_public=False, test=None, test_options=None, permutations=1, by_subject=False, public_name=None, state=None):
        # avoid _set_trf_options(**state) because _set_trf_options could catch invalid state
        # parameters like `scale`
        if metric and not FIT_METRIC_RE.match(metric):
            raise ValueError(f'metric={metric!r}')
        data = TestDims.coerce(data)

        if test:
            if state is None:
                state = {}
            if test is True:
                state['test'] = ''
            elif isinstance(test, str):
                state['test'] = test
            else:
                raise TypeError(f"test={test!r}")

        if state:
            self.set(**state)
        dstrf = self.get('inv') == 'dstrf'

        if dstrf:
            # make sure we are receiving default values
            assert basis == 0.050
            basis = None
            assert error == 'l1'
            error = None
            assert partitions is None
            partitions = -1
            assert mask is None
            assert delta == 0.005
            assert mindelta is None
            assert selective_stopping == 0
            assert data.source is True
            assert backward is False

        # model description
        if public_name is not None:
            assert is_public
            x_name = public_name
        else:
            x_name = self._x_desc(x, is_public)

        # TRF method
        trf_options = [] if dstrf else ['boosting']
        # basis
        if basis:
            trf_options.append(f'h{ms(basis)}')
        if error:
            trf_options.append(error)
        # cross-validation
        if partitions is None:
            trf_options.append('seg')
        elif partitions > 0:
            trf_options.append(f'{partitions}ptns')
        elif partitions < 0:
            trf_options.append(f'con{-partitions}ptns')
        # backward model
        if backward:
            trf_options.append('backward')
        # filter regressors
        if filter_x:
            trf_options.append('filtx')
        # delta
        assert 0. < delta < 1.
        if delta != 0.005 or mindelta is not None:
            if delta != 0.005:
                delta = str(delta)[2:]
            else:
                delta = ''
            if mindelta is not None:
                assert 0. < mindelta < delta
                mindelta = '>' + str(mindelta)[2:]
            else:
                mindelta = ''
            trf_options.append(delta + mindelta)
        if selective_stopping:
            assert isinstance(selective_stopping, int)
            trf_options.append(f'ss{selective_stopping}')
        if cv:
            trf_options.append('cv')
        if scale is not None:
            assert scale in ('original',)
            trf_options.append(scale)
        if smooth_source:
            mm = smooth_source * 1000.
            assert int(mm) == mm
            assert mm < 50.
            trf_options.append(f"s{int(mm)}mm")
        if smooth_time:
            trf_options.append(smooth_time if isinstance(smooth_time, str) else f"s{ms(smooth_time)}")
        # mask
        src = self.get('src')
        if mask:
            if not isinstance(mask, str):
                raise TypeError(f"mask={mask!r}")
            elif data.source is not True:
                raise ValueError(f"mask={mask!r} with data={data.string!r}")
            elif src.startswith('vol'):
                raise ValueError(f"mask={mask!r} with src={src!r}")
        else:
            assert mask is None
            if not dstrf and data.source is True and not src.startswith('vol'):
                raise ValueError(f"mask={mask!r} with src={src!r}")

        options = [x_name]
        # whether TRF or test (for backwards compatibility)
        if is_group_result:
            folder = trf_options
        else:
            options.extend(trf_options)
            if mask:
                options.insert(0, mask)
            folder = ()

        if metric:
            options.append(metric)
        if test_options:
            if isinstance(test_options, str):
                options.append(test_options)
            else:
                options.extend(test_options)
        if permutations != 1:
            options.append(f'{permutations}-pmts')
        if by_subject:
            options.append('subjects')

        if samplingrate is None:
            epoch = self._epochs[self.get('epoch')]
            if epoch.samplingrate is None:
                raise NotImplementedError("Epoch.samplingrate")  # FIXME
            samplingrate = epoch.samplingrate

        self._set_analysis_options(data, False, False, pmin, tstart, tstop, None, mask, samplingrate, options, folder)

    def _parse_trf_test_options(self, test_options: FieldCode):
        # FIXME:  is invalid for group result which has different options order
        code = FieldCode.coerce(test_options)
        out = self._parse_test_options(code)
        # mask
        if code.lookahead_1 in self._parcs:
            out['mask'] = code.next()
        # model
        model = code.next()
        if code.lookahead_1.startswith('('):
            assert '$' not in model
            out['rand'] = code.next()
        elif '$' in model:
            model, rand = model.split('$')
            out['rand'] = '$' + rand
        out['model'] = model
        # trf-options
        out['trf_options'] = ' '.join(code)
        return out

    def _parse_trf_path(self, filename: str):
        """Parse a TRF filename into components

        Notes
        -----
        template is ``{trf-sdir}/{subject}/{analysis}/{epoch} {test_options}.pickle``
        """
        path = Path(filename)
        epoch, test_options = path.stem.split(' ', 1)
        code = FieldCode(test_options)
        out = self._parse_trf_test_options(code)
        out['subject'] = path.parent.parent.name
        out['analysis'] = path.parent.name
        out['epoch'] = epoch
        return out

    def _coerce_model(self, x: Union[str, Model]) -> Model:
        if isinstance(x, Model):
            return x
        elif x in self._structured_models:
            return self._structured_models[x].model
        return Model.from_string(x).initialize(self._structured_models)

    def _coerce_comparison(
            self,
            x: ComparisonArg,
            cv: bool,
    ) -> Union[Comparison, StructuredModel]:
        if isinstance(x, str):
            return Comparison.coerce(x, cv, self._structured_models)
        elif not isinstance(x, (StructuredModel, Comparison)):
            raise TypeError(f"x={x!r}: need comparison")
        return x

    def _x_desc(self, x, is_public=False):
        "Description for x"
        if isinstance(x, Model):
            if not x:
                return '0'
            elif is_public:
                return x.name
            elif x.sorted_key in self._model_names:
                return self._model_names[x.sorted_key]
            elif x.without_randomization.sorted_key in self._model_names:
                xrand = x.randomized_component
                xrand_desc = self._x_desc(xrand.without_randomization)
                rand = {term.shuffle_string for term in xrand.terms}
                if len(rand) != 1:
                    raise NotImplementedError(f"{len(rand)} randomization schemes in {x}")
                rand_desc = f'{xrand_desc}{rand.pop()}'
                if xrand == x:
                    return rand_desc
                base_name = self._model_names[x.without_randomization.sorted_key]
                return f'{base_name} ({rand_desc})'
            else:
                self._register_model(x)
                return self._x_desc(x)
        elif isinstance(x, Comparison):
            if is_public:
                return x.name
            else:
                return x.compose_name(self._x_desc)
        elif isinstance(x, StructuredModel):
            assert is_public  # all internal names should be Model-based
            if x in self._structured_model_names:
                return self._structured_model_names[x]
            elif x.public_name:
                return x.public_name
            else:
                raise RuntimeError(f"{x} has no public name")
        else:
            raise TypeError(f"x={x!r}")

    def load_model_test(
            self,
            x: ComparisonArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: bool = False,
            selective_stopping: int = 0,
            cv: bool = False,
            data: DataArg = DATA_DEFAULT,
            permutations: int = 1,
            metric: str = 'z',
            smooth: str = None,
            test: str = None,
            return_data: bool = False,
            pmin: PMinArg = 'tfce',
            xhemi: bool = False,
            xhemi_mask: bool = True,
            make: bool = False,
            **state,
    ):
        """Test comparing model fit between two models

        Parameters
        ----------
        ...
        xhemi
            Test between hemispheres.
        xhemi_mask
            When doing ``xhemi`` test, mask data with region that is significant
            in at least one hemisphere.
        permutations
            When testing against a partially permuted model, average the result
            of ``permutations`` different permutations as baseline model.
        metric
            Fit metric to use for test:

            - ``r``:   Pearson correlation
            - ``z``:   z-transformed correlation
            - ``r1``:  1 correlation
            - ``z1``:  z-transformed l1 correlation
            - ``residual``: Residual form model fit
            - ``det``: Proportion of the explained variability

        smooth : float
            Smooth data in space before test (value in [m] STD of Gaussian).
        ...
        make : bool
            If the test does not exists, make it (the default is to raise an
            IOError).
        ...
            State parameters.

        Returns
        -------
        ds : Dataset | dict (if return_data==True)
            Dataset with values of the test and baseline models that forms the
            basis of the test.
        res : NDTest | ResultCollection
            Test result.
        """
        data = TestDims.coerce(data, time=False)
        comparison = self._coerce_comparison(x, cv)

        # Load multiple tests for a comparison group
        if isinstance(comparison, StructuredModel):
            if state:
                self.set(**state)
            ress = {comp.test_term_name: self.load_model_test(comp, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, permutations, metric, smooth, test, return_data, pmin, xhemi, xhemi_mask, make) for comp in comparison.comparisons(cv)}
            if return_data:
                dss = {key: res[0] for key, res in ress.items()}
                ress = ResultCollection({key: res[1] for key, res in ress.items()})
                return dss, ress
            else:
                return ResultCollection(ress)

        if xhemi:
            if xhemi_mask:
                test_options = 'xhemi.05'
            else:
                test_options = 'xhemi'
        elif xhemi_mask is not True:
            raise ValueError(f"xhemi_mask={xhemi_mask!r}; parameter is invalid unless xhemi=True")
        else:
            test_options = None

        test_desc = True if test is None else test
        self._set_trf_options(comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, pmin=pmin, test=test_desc, smooth_source=smooth, metric=metric, is_group_result=True, test_options=test_options, permutations=permutations, state=state)
        dst = self.get('model-test-file', mkdir=True)
        dst = self._cache_path(dst)
        if self._result_file_mtime(dst, data):
            res = load.unpickle(dst)
            if data.source:
                update_subjects_dir(res, self.get('mri-sdir'), 1)
        else:
            res = None

        y, to_uv = FIT_METRIC_RE.match(metric).groups()
        if to_uv:  # make sure variables that don't affect tests are default
            if xhemi:
                raise ValueError(f"xhemi={xhemi!r} with metric={metric!r}")
            elif pmin != 'tfce':
                raise ValueError(f"pmin={pmin!r} with metric={metric!r}")

        if return_data or res is None:
            # load data
            group = self.get('group')
            vardef = None if test is None else self._tests[test].vars
            x1_permutations = permutations if comparison.x1.has_randomization else 1
            ds1 = self.load_trfs(group, comparison.x1, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make, vardef=vardef, permutations=x1_permutations)

            if comparison.x0.terms:
                ds0 = self.load_trfs(group, comparison.x0, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make, vardef=vardef, permutations=permutations)
                # restructure data
                assert np.all(ds1['subject'] == ds0['subject'])
                if test is None:
                    ds = combine((ds1['subject', ], ds0['subject', ]))
                else:
                    keep = tuple(k for k in ds1 if isuv(ds1[k]) and np.all(ds1[k] == ds0[k]))
                    ds = ds1[keep]
                dss = [ds1, ds0]
            else:
                ds0 = None
                ds = ds1
                dss = [ds1]

            if test is None:
                if xhemi:
                    test_obj = TTestRelated('hemi', 'lh', 'rh')
                elif ds0 is None:
                    test_obj = TTestOneSample(comparison.tail)
                else:
                    test_obj = TTestRelated('model', 'test', 'baseline', comparison.tail)
            else:
                test_obj = self._tests[test]

            # smooth data (before xhemi morph)
            if smooth:
                for ds_i in dss:
                    ds_i[y] = ds_i[y].smooth('source', smooth, 'gaussian')

            if xhemi:
                if ds0 is not None:
                    ds1[y] -= ds0[y]
                lh, rh = eelbrain.xhemi(ds1[y], parc=self._xhemi_parc())
                if test is None:
                    ds[y] = combine((lh, rh))
                    ds['hemi'] = Factor(('lh', 'rh'), repeat=ds1.n_cases)
                else:
                    ds[y] = lh - rh

                # mask
                if xhemi_mask:
                    parc = self._xhemi_parc()
                    with self._temporary_state:
                        base_res = self.load_model_test(comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, permutations, metric, smooth, test, pmin=pmin, make=make)
                    if isinstance(base_res, MultiEffectNDTest):
                        raise NotImplementedError("xhemi_mask for multi-effect tests")
                    mask_lh, mask_rh = eelbrain.xhemi(base_res.p <= 0.05, parc=parc)
                    np.maximum(mask_lh.x, mask_rh.x, mask_lh.x)
                    ds[y] *= mask_lh > .5
            elif ds0 is None:
                pass
            elif test is None:
                ds[y] = combine((ds1[y], ds0[y]))
                ds['model'] = Factor(('test', 'baseline'), repeat=ds1.n_cases)
            else:
                ds[y] = ds1[y] - ds0[y]

            # test
            if res is None:
                # test arguments
                kwargs = self._test_kwargs(10000, pmin, None, None, data, None)
                res = self._make_test(y, ds, test_obj, kwargs, to_uv=to_uv)
                save.pickle(res, dst)

            if return_data:
                return ds, res
        return res

    def _locate_model_test_trfs(self, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, permutations=1, existing=False, **state):
        """Find required jobs for a report

        Returns
        -------
        trf_jobs : list
            List of ``(path, state, args)`` tuples for missing TRFs.
        """
        if state:
            self.set(**state)

        if isinstance(x, StructuredModel):
            models = {m for comp in x.comparisons(cv) for m in comp.models}
        else:
            models = x.models

        missing = []
        for model in models:
            missing.extend(
                self._locate_missing_trfs(model, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, False, permutations, existing))
        return missing

    def make_model_test_report(self, x, tstart=0, tstop=0.5, basis=0.050, error='l1', partitions=None, samplingrate=None, mask=None, delta=0.005, mindelta=None, filter_x=False, selective_stopping=0, cv=False, data=DATA_DEFAULT, permutations=1, metric='z', smooth=None, surf=None, views=None, make=False, path_only=False, public_name=None, test=True, by_subject=False, **state):
        """Generate report for model comparison

        Parameters
        ----------
        ...
        by_subject : bool
            Generate a report with each subject's data.

        Returns
        -------
        path : str
            Path to thre report (only returned with ``path_only=True`` or if the
            report is newly created.
        """
        data = TestDims.coerce(data)
        if data.source is not True:
            raise NotImplementedError("Model-test report for data other than source space")
        x = self._coerce_comparison(x, cv)
        self._set_trf_options(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, metric=metric, smooth_source=smooth, is_group_result=True, is_public=True, test=test, permutations=permutations, by_subject=by_subject, public_name=public_name, state=state)
        dst = self.get('model-report-file', mkdir=True)
        if path_only:
            return dst
        elif exists(dst):
            return
        self._log.info("Make TRF-report: %s", relpath(dst, self.get('model-res-dir')))

        ds, res = self.load_model_test(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, permutations, metric, smooth, test, True, 'tfce', make=make)

        if isinstance(x, StructuredModel):
            comparisons = x.comparisons(cv)
            dss, ress = ds, res
            ds = dss[comparisons[0].test_term_name]
            res = ress[comparisons[0].test_term_name]
        else:
            comparisons = (x,)
            dss = {x.test_term_name: ds}
            ress = {x.test_term_name: res}

        if data.source:
            inv = self.get('inv')
            is_vector_data = inv.startswith('vec')
        else:
            is_vector_data = False

        # Report
        if public_name is None:
            public_name = self._x_desc(x, is_public=True)
        report = Report(public_name)
        report.add_paragraph(self._report_methods_brief(dst))

        if is_vector_data:
            if by_subject:
                raise NotImplementedError
            for comp in comparisons:
                section = trf_report.vsource_tfce_result(ress[comp.test_term_name], comp.test_term_name, f"{comp.x1_only} > {comp.x0_only}")
                report.append(section)
        else:
            surfer_kwargs = self._surfer_plot_kwargs(surf, views)
            if by_subject:
                subjects, diffs = difference_maps(dss)
                if 'hemi' in surfer_kwargs:
                    hemis = (surfer_kwargs.pop('hemi'),)
                else:
                    hemis = ('lh', 'rh')

                for hemi in hemis:
                    section = report.add_section(hemi)
                    brain = plot.brain.brain(ds[metric].source, w=220, h=150, hemi=hemi, **surfer_kwargs)
                    # brain.set_parallel_view(*BRAIN_VIEW[1:])
                    for x_ in comparisons:
                        subsection = section.add_section(x_.test_term_name)
                        row = []
                        for subject, dmap in zip(subjects, diffs[x_.test_term_name, hemi]):
                            brain.add_ndvar(dmap, remove_existing=True)
                            brain.add_text(0, 0, subject, 'subject', (0, 0, 0), font_size=30, justification='left')
                            brain.texts_dict['subject']['text'].property.font_size = 28
                            row.append(brain.image())
                        subsection.append(fmtxt.Figure(row))
                    brain.close()
                report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
                report.save_html(dst)
                return
            brain = None
            for comp in comparisons:
                section, brain = trf_report.source_tfce_result(ress[comp.test_term_name], surfer_kwargs, comp.test_term_name, f"{comp.x1_only} > {comp.x0_only}", brain)
                report.append(section)
            brain.close()

        # Info section
        if test is True:
            test = TTestRelated('model', 'test', 'baseline', comparisons[0].tail)
        sec = report.add_section("Info")
        info = self._report_test_info(sec, ds, test, res, data, model=False)
        # info:
        info.add_item(f"Mask: {mask}")
        # Info: model
        model_info = List("Predictor model")
        if isinstance(x, StructuredModel):
            model_info.add_item("Incremental model improvement for each term")
            model_info.add_item(x.public_name)
        elif isinstance(x, Comparison):
            if x.common_base:
                model_info.add_item("Common base:  " + x.common_base.name)
                model_info.add_item("Test model:  + " + x.x1_only.name)
                model_info.add_item("Baseline model:  + " + x.x0_only.name)
            else:
                model_info.add_item("Test model:  " + x.x1.name)
                model_info.add_item("Baseline model:  " + x.x0.name)
        if permutations > 1:
            model_info.add_item(f"Tests against {permutations} permutations.")
        info.add_item(model_info)
        # Info: reverse correlation method
        trf_info = List("TRF estimation using boosting")
        trf_info.add_item(f"TRF {ms(tstart)} - {ms(tstop)} ms at {ds.info['samplingrate']:g} Hz")
        if basis:
            trf_info.add_item(f"Basis of {ms(basis)} ms Hamming windows")
        trf_info.add_item(f"Error function: {error}")
        trf_info.add_item(f"∆ = {delta}")
        if mindelta is not None:
            trf_info.add_item(f"min-∆ = {mindelta}")
        if ds.info['partitions'] == -1:
            trf_info.add_item(f"Fitted to continuous data with 10 partitions")
        else:
            trf_info.add_item(f"Fitted to segmented data with {ds.info['partitions']} partitions")
        if filter_x:
            trf_info.add_item("Regressors filtered like data")
        if selective_stopping:
            trf_info.add_item(f"Selective stopping after {n_of(selective_stopping, 'failure')}")
        info.add_item(trf_info)
        # Signature
        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst)
        return dst

    def invalidate(self, regressor):
        """Remove cache and result files when input data becomes invalid

        Parameters
        ----------
        regressor : str
            Regressor that became invalid; can contain ``*`` and ``?`` for
            pattern matching.
        """
        files = set()  # avoid duplicate paths when model name contains regressor name

        # patterns
        reg_re = fnmatch.translate(regressor)
        reg_re_term = re.compile(rf"^{reg_re}$")

        # find all named models that contain term
        models = []  # single-term model
        for name, model in self._named_models.items():
            if any(reg_re_term.match(term.code) for term in model.terms_without_randomization):
                models.append(name)
                files.update(self._find_model_files(name, trfs=True, tests=True))

        # cached regressor files
        cache_dir = self.get('predictor-cache-dir', mkdir=True)
        files.update(glob(join(cache_dir, f'*|{regressor} *.pickle')))

        if not files:
            print("No files affected")
            return

        while True:
            command = ask(f"Invalidate {regressor} regressor, deleting {len(files)} files?", {'yes': 'delete files', 'show': 'list files to be deleted'}, allow_empty=True)
            if command == 'yes':
                for path in files:
                    os.remove(path)
            elif command == 'show':
                print(f"Models: {', '.join(models)}")
                paths = sorted(files)
                prefix = os.path.commonprefix(paths)
                print(f"Files in {prefix}:")
                for path in paths:
                    print(relpath(path, prefix))
                continue
            return

    # Source estimation
    ###################
    def load_psf(self, mask=True, **state):
        """Load inverse point spread function
        
        brain = plot.brain.brain(psf.source, mask=False, hemi='lh')
        brain.add_ndvar(psf[:, 'transversetemporal-lh'][:, 0])
        """
        if isinstance(mask, str):
            state['parc'] = mask
            mask = True
        if state:
            self.set(**state)
        inv_op = self.load_inv(ndvar=True, mask=mask)
        inv_op.source.subject = 'fsaverage'
        inv_op = rename_dim(inv_op, 'source', 'source_to')
        fwd_op = self.load_fwd(ndvar=True, mask=mask).sub(sensor=inv_op.sensor)
        fwd_op.source.subject = 'fsaverage'
        fwd_op = rename_dim(fwd_op, 'source', 'source_from')
        psf = inv_op.dot(fwd_op)
        return psf

    # Long filenames
    ################
    def _cache_path(self, path):
        dirname, basename = os.path.split(path)
        if len(basename) < 256:
            return path
        raise RuntimeError("Path too long: %s" % (path,))
        # is this really necessary...?
        registry_path = join(self.get('cache-dir'), 'shortened_paths.pickle')
        if exists(registry_path):
            registry = load.unpickle(registry_path)
        else:
            registry = {}

        if basename not in registry:
            name, ext = splitext(basename)
            prefix = name[:220]
            dst = '%s shortened-%%i%s' % (prefix, ext)
            values = registry.values()
            i = 0
            while dst % i in values:
                i += 1
            registry[basename] = dst % i
            save.pickle(registry, registry_path)
        return join(dirname, registry[basename])

    def remove_model(self, model):
        """Remove a named model and delete all associated files

        See Also
        --------
        .clean_models
        .show_models
        """
        if model in self.models:
            raise ValueError(f"{model!r} is an explicitly defined model; remove it from .models")
        self._remove_model(model)

    def show_contamination(self, threshold=2e-12, separate=False, absolute=False, samplingrate=None, asds=False, **state):
        """Table of data exceeding threshold in epochs

        Determine the amount of time during which the absolute value from at
        least one sensor exceeds ``threshold``.

        Parameters
        ----------
        threshold : scalar
            Absolute threshold.
        separate : bool
            Include separate statistics for each epoch (default False).
        absolute : bool
            List absolute number of samples exceeding threshold (default is
            the percentage).
        samplingrate : int
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        asds : bool
            Return results as :class:`Dataset` instead of a table.
        ...
            State parameters.

        Returns
        -------
        table : fmtxt.Table | Dataset
            Result, either as table for display (default) or, with
            ``asds=True``, as  :class:`Dataset`.
        """
        if absolute:
            def agg(x):
                return x.sum()
        else:
            def agg(x):
                return int(round(x.mean() * 100.))

        lines = []
        for subject in self.iter(**state):
            ds = self.load_epochs(samplingrate=samplingrate)
            meg_abs = ds['meg'].extrema('sensor').abs()
            line = [subject, agg(meg_abs > threshold)]
            if separate:
                line.extend(agg(epoch > threshold) for epoch in meg_abs)
            lines.append(line)
        headings = ['Subject', 'Total']
        if separate:
            n_entries = max(map(len, lines))
            n_epochs = n_entries - 2
            headings.extend('Ep %i' % i for i in range(n_epochs))
            for line in lines:
                for _ in range(n_entries - len(line)):
                    line.append(np.nan)
        else:
            n_epochs = 0
        if asds:
            return Dataset.from_caselist(headings, lines)

        table = Table('lr' + 'r' * n_epochs)
        table.cells(*headings)
        table.midrule()
        for line in lines:
            table.cells(*line)
        return table

    def show_cached_trfs(self, model=None, keys=('analysis', 'epoch', 'time_window', 'samplingrate', 'model', 'mask')):
        """List cached TRFs and how much space they take

        Parameters
        ----------
        model : str
            String to fnmatch the model.
        keys : tuple of str
            Keys which to use to group TRFs in the table.

        See Also
        --------
        .show_models

        Notes
        -----
        To delete TRFs corresponding to a specific model, use, for example::

            e.rm('trf-file', True, test_options='* model *')

        Note that some fields are embedded, e.g. ``raw`` in ``analysis``, so to
        delete files with ``raw='1-8'``, use::

            e.rm('trf-file', True, test_options='* model *', analysis='1-8 *')

        """
        ns = defaultdict(lambda: 0)
        sizes = defaultdict(lambda: 0.)  # in bytes
        for path in self.glob('trf-file', True):
            properties = self._parse_trf_path(path)
            if not model or fnmatch.fnmatch(properties['model'], model):
                key = tuple(properties.get(k, '') for k in keys)
                ns[key] += 1
                sizes[key] += os.stat(path).st_size
        sorted_keys = sorted(ns)
        t = fmtxt.Table('l' * len(keys) + 'rr')
        t.cells(*keys, 'n', 'size (MB)')
        t.midrule()
        for key in sorted_keys:
            t.cells(*key)
            t.cell(ns[key])
            size_mb = round(sizes[key] / 1e6, 1)
            t.cell(size_mb)
        return t

    def show_comparison_terms(self, comparison: str, cv: bool = False):
        """Generate a table comparing the terms in the two models"""
        comp = self._coerce_comparison(comparison, cv)
        return comp.term_table()

    def show_models(self, term=None, stim=True, rand=True, model=None, sort=False, files=False):
        """List models that contain a term that matches ``term``

        Parameters
        ----------
        term : str
            Fnmatch pattern for a terms.
        stim : bool
            Also include terms with a stimulus prefix.
        rand : bool
            Also show models that contain ``term`` randomized.
        model : str
            Pattern to display only certain models.
        sort : bool
            Sort terms (default False).
        files : bool
            List the number of files associated with the model.

        See Also
        --------
        .remove_model
        .show_cached_trfs

        Notes
        -----
        Initial column contains ``*`` for models explicitly defined in
        :attr:`.models`.
        """
        if term is None:
            pattern = None
        else:
            pattern = fnmatch.translate(term)
            if stim:
                pattern = r'(\w+\|)?' + pattern
            if rand:
                pattern += r'(\$.*)?'
            pattern = re.compile(pattern)
        model_pattern = model

        columns = 'lll'
        if files:
            columns += 'rr'
        t = fmtxt.Table(columns)
        t.cells('.', 'Name', 'Terms')
        if files:
            t.cells('trfs', 'tests')
        t.midrule()
        for name, model in self._named_models.items():
            if pattern is not None:
                if not any(pattern.match(t.string) for t in model.terms):
                    continue
            if model_pattern is not None:
                if not fnmatch.fnmatch(name, model_pattern):
                    continue
            t.cell('*' if name in self.models else '')
            t.cell(name)
            if sort:
                t.cell(model.sorted_key)
            else:
                t.cell(model.name)
            if files:
                n = len(self._find_model_files(name, trfs=True))
                t.cell(n or '')
                n = len(self._find_model_files(name, tests=True))
                t.cell(n or '')
        return t
