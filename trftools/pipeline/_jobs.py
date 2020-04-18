from argparse import ArgumentParser
from operator import itemgetter
import os
from os.path import relpath, dirname
import sys
from typing import Any, Dict, Union
import webbrowser

from eelbrain import fmtxt, save
from eelbrain._experiment.test_def import TestDims

from ._model import StructuredModel, Model, ModelArg


def make_jobs(job_file, make_trfs=False, open_in_browser=False):
    for job in read_job_file(job_file):
        if isinstance(job, TRFsJob):
            if not make_trfs:
                continue
        elif make_trfs:
            job.options['make'] = True
        path = job.execute()
        if path and open_in_browser:
            webbrowser.open(f"file://{path}")


class TRFJob:
    """Jopb for a single TRF"""
    __slots__ = ('experiment', 'path', 'state', 'args', 'desc')

    def __init__(self, experiment, path, state, args):
        self.experiment = experiment
        self.path = path
        self.state = state
        self.args = args
        self.desc = f"{experiment.__class__.__name__} {relpath(path, experiment.get('trf-sdir'))}"

    def __repr__(self):
        return f"<TRFJob: {self.desc}>"

    def generate_job(self):
        self.experiment._restore_state(self.state)
        return self.experiment._trf_job(*self.args)

    def _execute(self):
        job = self.generate_job()
        if job:
            save.pickle(job(), self.path)


class Job:
    """Baseclass for multi-TRF jobs (TRFs or model-comparison)"""
    def __init__(
            self,
            experiment,
            model: ModelArg,
            priority: bool,
            options: Dict[str, Any],
            postfit: ModelArg = False,
            public_name: str = None,
            report: bool = False,
    ):
        # validate input
        from ._experiment import TRFExperiment
        if not isinstance(experiment, TRFExperiment):
            raise TypeError(f"experiment={experiment!r}")

        if 'data' in options:
            options['data'] = TestDims.coerce(options['data'])

        self.model = model
        self.report = report
        self.experiment = experiment
        self.options = options
        self.priority = priority
        self.postfit = postfit
        self.public_model_name = public_name
        self.model_name = public_name or experiment._x_desc(model, True)

        # to be initialized
        self.trf_jobs = None
        self.missing_trfs = None
        self.test_path = None

    def __repr__(self):
        args = [self.model_name]
        args.extend(f'{k}={v!r}' for k, v in self.options.items())
        return f"<{self.__class__.__name__}: {', '.join(args)}>"

    def is_same(self, other: 'Job'):
        return (
            type(self) is type(other) and
            self.model == other.model and
            self.report == other.report and
            self.experiment.__class__.__name__ == other.experiment.__class__.__name__ and
            self.options == other.options and
            self.postfit == other.postfit)

    def init_test_path(self):
        raise NotImplementedError

    def init_sub_jobs(self):
        self.trf_jobs = self.generate_trf_jobs()
        self.missing_trfs = {job.path for job in self.trf_jobs}

    def _init_trf_jobs(self, existing: bool = False):
        raise NotImplementedError

    def execute(self):
        self.experiment.reset()
        return self._execute()

    def _execute(self):
        raise NotImplementedError

    def has_followup_jobs(self):
        raise NotImplementedError

    def get_followup_jobs(self, log):
        raise NotImplementedError

    def generate_trf_jobs(self, existing: bool = False):
        """List of TRFJob

        Parameters
        ----------
        existing : bool
            Include jobs for TRFs that are already cached in pickle files
            (default ``False``).
        """
        self.experiment.reset()
        return [TRFJob(self.experiment, *args) for args in self._init_trf_jobs(existing)]


class FuncJob:

    def __init__(self, name, jobs, priority=False):
        self.name = name
        self.jobs = jobs
        self.priority = priority
        self._i = -1

    def __next__(self):
        out = next(self.jobs)
        self._i += 1
        return out

    @property
    def desc(self):
        return f"{self.name}-{self._i}"


class TRFsJob(Job):
    """Job for group of TRFs

    Parameters
    ----------
    model : str
        Model or regressors.
    report : bool
        Schedule a model-test report.
    experiment : Experiment
        The experiment instance providing access to the data.
    reduce_model : bool
        Reduce the model until it only contains predictors significant at the
        .05 level.
    parent : Job
        Parent job (for reduced model jobs).
    priority : bool
        Insert job at the beginning of the queue (default ``False``).
    postfit : Model | bool
        Component of ``x`` to post-fit.
    reduction_tag : str
        Tag to use for reduced models (to distinguish different reduction
        algorithms, default ``'red'``).
    ...
        Model-test parameters.
    """
    _prefit_done = False  # mark whether this is the first or second stage

    def __init__(
            self,
            model: ModelArg,
            experiment=None,
            priority: bool = False,
            postfit: Union[ModelArg, bool] = False,
            **options,
    ):
        model = experiment._coerce_model(model)
        Job.__init__(self, experiment, model, priority, options, postfit)

    def is_same(self, other: 'TRFsJob'):
        return (
            Job.is_same(self, other) and
            self._prefit_done == other._prefit_done)

    def init_test_path(self):
        pass

    def _init_trf_jobs(self, existing: bool = False):
        if self.postfit and not self._prefit_done:
            if self.postfit is True:
                terms = [term.string for term in self.model.terms]
            else:
                terms = [self.postfit]
            out = []
            for term in terms:
                model = self.model - self.experiment._coerce_model(term)
                out.extend(self.experiment._locate_missing_trfs(model, existing=existing, **self.options))
            return out
        return self.experiment._locate_missing_trfs(self.model, existing=existing, **self.options, postfit=self.postfit)

    def _execute(self):
        self.init_sub_jobs()
        for job in self.trf_jobs:
            job._execute()
        for job in self.get_followup_jobs():
            job.execute()

    def has_followup_jobs(self):
        return self.postfit and not self._prefit_done

    def get_followup_jobs(self, log=None):
        if self.postfit and not self._prefit_done:
            job = TRFsJob(self.model, self.experiment, self.priority, self.postfit, **self.options)
            job._prefit_done = True
            return [job]
        else:
            return []


class ModelJob(Job):
    """Job for model-comparison

    Parameters
    ----------
    model : str
        StructuredModel or comparison (see :class:`TRFExperiment` module docstring).
    report : bool
        Schedule a model-test report.
    experiment : Experiment
        The experiment instance providing access to the data.
    reduce_model : bool
        Reduce the model until it only contains predictors significant at the
        .05 level.
    parent : Job
        Parent job (for reduced model jobs).
    priority : bool
        Insert job at the beginning of the queue (default ``False``).
    postfit : Model | bool
        Component of ``x`` to post-fit.
    reduction_tag : str
        Tag to use for reduced models (to distinguish different reduction
        algorithms, default ``'red'``).
    ...
        Model-test parameters.
    """
    def __init__(self, model, experiment=None, report=True, reduce_model=False, parent=None, priority=False, postfit=False, reduction_tag='red', metric='z', smooth=False, **options):
        assert postfit is False
        model = experiment._coerce_comparison(model)
        if isinstance(reduce_model, float):
            assert 0. < reduce_model < 1.
        elif not isinstance(reduce_model, bool):
            raise TypeError(f"reduce_model={reduce_model!r}")
        if reduce_model and not isinstance(model, StructuredModel):
            raise ValueError("reduce_model requires incremental model-comparison as base")

        # spacial description for reduced models
        if parent is not None:
            i = 1
            current = parent
            while current.parent is not None:
                i += 1
                current = current.parent
            public_name = '%s-%s%i' % (experiment._x_desc(current.model, True), reduction_tag, i)
        else:
            public_name = None

        Job.__init__(self, experiment, model, priority, options, postfit, public_name, report)
        self._test_options = {'metric': metric, 'smooth': smooth}
        self._reduction_tag = reduction_tag
        self.reduce_model = reduce_model
        self.parent = parent
        self._reduction_results = []

    def is_same(self, other: 'ModelJob'):
        return (
            Job.is_same(self, other) and
            self._test_options == other._test_options)

    def init_test_path(self):
        self.experiment.reset()
        self.test_path = self.experiment.make_model_test_report(self.model, public_name=self.public_model_name, path_only=True, **self.options, **self._test_options)

    def _init_trf_jobs(self, existing: bool = False):
        return self.experiment._locate_model_test_trfs(self.model, existing=existing, **self.options)

    def _execute(self):
        job = self.reduced_model_job()
        if job:
            job.execute()
            self._reduction_results.extend(job._reduction_results)
        elif self.report:
            return self.experiment.make_model_test_report(self.model, public_name=self.public_model_name, **self.options, **self._test_options)

    def has_followup_jobs(self):
        return self.reduce_model

    def get_followup_jobs(self, log):
        try:
            job = self.reduced_model_job()
        except:
            log.exception("Generating reduced model job for %s", self.test_path or self.public_model_name)
            job = None
        else:
            if job is None:
                log.info("No further reduction of %s", self.test_path or self.public_model_name)
            else:
                log.info("Reduced model of %s", self.test_path or self.public_model_name)
        if job is None:
            return []
        else:
            return [job]

    def make_test_report(self):
        assert self.test_path is not None, "Job has no test-report"
        self.experiment.reset()
        self.experiment.make_model_test_report(self.model, public_name=self.public_model_name, **self.options, **self._test_options)

    def reduced_model_job(self):
        if not self.reduce_model:
            return
        p_threshold = .05 if self.reduce_model is True else self.reduce_model
        self.experiment.reset()
        ress = self.experiment.load_model_test(self.model, **self.options, **self._test_options)
        self._reduction_results.append(ress)
        # no terms left
        if len(ress) == 0:
            return
        # find term to remove
        pmins = [(term, res.p.min()) for term, res in ress.items()]
        least_term, pmax = max(pmins, key=itemgetter(1))
        if pmax <= p_threshold:
            return  # all regressors are significant
        # check for ambiguity
        pmins = [term for term, p in pmins if p == pmax]
        if len(pmins) > 1:
            tmaxs = [(term, ress[term].t.max()) for term in pmins]
            least_term, tmin = min(tmaxs, key=itemgetter(1))
        # remove term
        model = self.model.without(least_term)
        return ModelJob(model, self.experiment, self.report, self.reduce_model, self, self.priority, self.postfit, self._reduction_tag, **self._test_options, **self.options)

    def reduction_table(self, labels=None, vertical=False, title=None, caption=None):
        """Table with steps of model reduction

        Parameters
        ----------
        labels : dict {str: str}
            Substitute new labels for predictors.
        vertical : bool
            Orient table vertically.
        title : text
            Title for the table.
        caption : text
            Caption for the table.
        """
        if not self._reduction_results:
            self.execute()
        if labels is None:
            labels = {}
        n_steps = len(self._reduction_results)
        # find terms
        terms = []
        for ress in self._reduction_results:
            terms.extend(term for term in ress.keys() if term not in terms)
        n_terms = len(terms)
        # cell content
        cells = {}
        for x in terms:
            for i, ress in enumerate(self._reduction_results):
                if x in ress:
                    res = ress[x]
                    pmin = res.p.min()
                    t_cell = fmtxt.stat(res.t.max(), stars=pmin)
                    p_cell = fmtxt.p(pmin)
                else:
                    t_cell = p_cell = ''
                cells[i, x] = t_cell, p_cell

        if vertical:
            t = fmtxt.Table('ll' + 'l' * n_terms, title=title, caption=caption)
            t.cells('Step', '')
            for x in terms:
                t.cell(labels.get(x, x))
            t.midrule()
            for i in range(n_steps):
                t_row = t.add_row()
                p_row = t.add_row()
                t_row.cells(i + 1, fmtxt.symbol('t', 'max'))
                p_row.cells('', fmtxt.symbol('p'))
                for x in terms:
                    t_cell, p_cell = cells[i, x]
                    t_row.cell(t_cell)
                    p_row.cell(p_cell)
        else:
            t = fmtxt.Table('l' + 'rr' * n_steps, title=title, caption=caption)
            t.cell()
            for _ in range(n_steps):
                t.cell(fmtxt.symbol('t', 'max'))
                t.cell(fmtxt.symbol('p'))
            t.midrule()
            for x in terms:
                t.cell(labels.get(x, x))
                for i in range(n_steps):
                    t.cells(*cells[i, x])
        return t


def split_lines(string):
    return (line for line in (line.strip() for line in string) if line and not
    line.startswith('#'))


def read_job_file(filename):
    """Read file with job definitions

    Returns
    -------
    jobs : list of Job
        All jobs found in the file.
    """
    with open(filename) as fid:
        text = fid.read()
    code = compile(text, filename, 'exec')
    namespace = {}
    # execute code allowing local imports
    filename = os.path.abspath(filename)
    file_dir = dirname(filename)
    if file_dir:
        sys.path.insert(0, file_dir)
    try:
        exec(code, namespace)
    finally:
        if file_dir:
            sys.path.remove(file_dir)
    # retrieve jobs
    jobs = list(namespace.get('JOBS', ()))
    if not jobs:
        raise RuntimeError(f"No JOBS in file {filename}")
    return jobs


def make_jobs_command():
    """Command-line command

    Usage::

        $ trf-tools-make-jobs jobs.py
    """
    argparser = ArgumentParser(description="TRF-Tools make-jobs")
    argparser.add_argument('job_file')
    argparser.add_argument('--make-trfs', action='store_true', help="Compute TRFs if not already present")
    argparser.add_argument('--open', action='store_true', help="Open new reports in browser")
    args = argparser.parse_args()
    make_jobs(args.job_file, args.make_trfs, args.open)
