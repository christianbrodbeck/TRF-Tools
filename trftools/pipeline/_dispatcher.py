# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Connect Experiment and Serve

Notes
=====

Experiment:
    .locate_trf:  return filename & whether it exists or instruction
                  (func, kwargs) to make it
    .make_trf:    makes trf-file based on filename and kwargs or sends to
                  dispatcher
    .locate_trfs: find list of TRFs that still have to be made
    .make_trf_report: makes report assuming filenames exist

Dispatcher:
    .add_job    - scans for which trfs still have to be made (requirement)
                - saves {report: requirement}
                - stores requirement jobs

    threaded:   - adds jobs to server (loader thread)
                - gets results from server, checks off requirements, runs
                  make_report once requirements are fulfilled
"""
import atexit
from collections import Counter
import fnmatch
import inspect
import logging
from collections import deque
from os.path import commonprefix, exists
from queue import Queue, Empty
import sys
from threading import Lock, Thread
from time import sleep, time

from eelbrain import fmtxt, boosting
from eelbrain._utils.com import Notifier
from eelbrain._utils import ask
from eelfarm.server import JobServer, JobServerTerminated

from ._jobs import FuncIterJob, read_job_file


MIN_IO_CYCLE_TIME = 5  # I/O thread
TRF_IRRELEVANT_STATE_KEYS = ('group', 'test', 'test_desc', 'model')


def dict_difference(name1, d1, name2, d2):
    if d1 == d2:
        return {}
    diff = {}
    for k in set(d1).union(d2):
        if k not in d1:
            diff[k] = f"only in {name2} ({d2[k]!r})"
        elif k not in d2:
            diff[k] = f"only in {name1} ({d1[k]!r})"
        elif d1[k] != d2[k]:
            diff[k] = f"{name1} != {name2} ({d1[k]!r} != {d2[k]!r})"
    return diff


def assert_trf_jobs_equal(new_job, old_job):
    "List of problem desciptions (or None if there is none)"
    if new_job == old_job:
        return
    problems = []
    for name, new, old in zip(('fields', 'field_values', 'params'), new_job.state, old_job.state):
        if name == 'field_values':
            continue
        s_diff = dict_difference('new', new, 'old', old)
        if s_diff:
            for k, v in s_diff.items():
                if k in TRF_IRRELEVANT_STATE_KEYS:
                    continue
                problems.append(f" state({name}) {k}: {v}")
    if new_job.args != old_job.args:
        arg_names = list(inspect.signature(boosting).parameters)
        new = dict(zip(arg_names, new_job.args))
        old = dict(zip(arg_names, old_job.args))
        s_diff = dict_difference('new', new, 'old', old)
        for key, desc in s_diff.items():
            if key == 'partition_results':
                if old[k]:
                    continue
            problems.append(f"boosting argument {key}: {desc}")
    return problems


def print_traceback(exc_info):
    # call_pdb does not work properly from a different thread. Instead, can
    # I store the traceback and call ipdb form the main thread, e.g. with
    # Dispatcher.debug()?
    if not hasattr(print_traceback, 'TRACEBACK_PRINTER'):
        from IPython.core.ultratb import VerboseTB
        print_traceback.TRACEBACK_PRINTER = VerboseTB()
    print_traceback.TRACEBACK_PRINTER(*exc_info)


class Dispatcher:
    """Dispatch jobs to Eelfarm server"""

    def __init__(
            self,
            host: str = None,
            port: int = 8000,
            job_queue_length: int = 2,
            notify: str = False,
    ):
        self.server = JobServer(host, port, job_queue_length)
        self.e_lock = Lock()  # access to experiment
        self.logger = logging.getLogger('eelfarm.dispatcher')
        # queues
        self._request_queue = Queue()
        self._user_jobs = []  # jobs added by the user
        self._report_jobs = {}  # jobs with unique target report file
        self._trf_job_queue = deque()
        self._trf_jobs = {}
        self._job_finalize_queue = deque()  # once all TRFs are present
        self._report_queue = Queue()  # schedule for report (manual)
        # flags
        self._auto_make_reports = False
        self._shutdown = False
        self.terminated = False
        # initiate threads
        self._thread = Thread(target=self._local_io, name="dispatcher")
        self._requeue_thread = Thread(target=self._requeue_failed_jobs, name="requeuer")
        if notify:
            self._notifier = Notifier(notify, 'Dispatcher')
        else:
            self._notifier = None
        atexit.register(self.shutdown, True)
        # make client methods available
        self.show_workers = self.server.show_workers
        self.blacklist_worker = self.server.blacklist_worker
        self.shutdown_worker = self.server.shutdown_worker

    def start(self):
        self.server.start()
        self._thread.start()
        self._requeue_thread.start()
        if self._notifier:
            self.logger.info("Notification to %s", self._notifier.to)

    def __repr__(self):
        if self.terminated:
            info = "successfully terminated"
        elif self._shutdown:
            info = f"shutting down, waiting for {self.server.n_pending_jobs()} pending TRFs..."
        else:
            items = [f"{len(self._trf_jobs)} pending TRFs"]
            n_reports = self._report_queue.qsize()
            if n_reports:
                items.append(f"{n_reports} reports ready")
            info = ', '.join(items)
        return f"<Dispatcher: {info}>"

    def add_jobs_from_file(self, filename, priority=False):
        """Add jobs from a job-file to the queue

        Parameters
        ----------
        filename : str
            Path to the job file.
        priority : bool
            Insert the jobs at the beginning of the queue (default ``False``).
        """
        if self._shutdown:
            raise RuntimeError("Dispatcher is shutting down.")
        n = 0
        with self.e_lock:
            for job in read_job_file(filename):
                if priority:
                    job.priority = True
                self._request_queue.put(job)
                n += 1
        self.logger.info("%i jobs requested", n)

    def add_jobs_from_iter(self, name, job_factory, priority=False):
        """Add jobs by providing an iterator over ``(path, func)`` pairs

        Parameters
        ----------
        name : str
            Name by which the job will be known.
        job_factory : iterator over (path_like, callable)
            Iterator over ``(path, job_loader)`` tuples. A ``job_loader`` is a
            function which will load the required data and return the ``job``.
            The ``job`` itself is a function, whose return value will be saved at
            ``path`` (see example). Make sure that most of the data is not
            loaded until ``job`` is called.
        priority : bool
            Insert the jobs at the beginning of the queue (default ``False``).

        Examples
        --------
        Generate a job for each of several subjects::

            from functools import partial
            import os
            from eelbrain import boosting
            import trftools


            def load_job(subject):
                y = load_data_for(subject)
                x = load_predictors_for(subject)
                return partial(boosting, y, x, tstart=0, tstop=0.500)

            def job_factory():
                for subject in range(1, 10):
                    path = f'save/to-{subject}.pickle'
                    if os.path.exists(path):
                        continue
                    job_loader = partial(load_job, subject)
                    yield path, job_loader

            dispatcher = trftools.start_dispatcher()
            dispatcher.add_jobs_from_iter('my_job', job_factory())

        An example that works, turning strings into uppercase::

            def load_job(string):
                return partial(str.upper, string)

            def job_factory():
                for string in ['test', 'this']:
                    path = f'{string}.pickle'
                    if os.path.exists(path):
                        continue
                    job_loader = partial(load_job, string)
                    yield path, job_loader

            dispatcher.add_jobs_from_iter('my_job', job_factory())

        """
        self._request_queue.put(FuncIterJob(name, job_factory, priority))

    def _local_io(self):
        n_exceptions = n_trf_exceptions = 0
        while True:
            cycle_start_time = time()

            # schedule new jobs (on the same thread to make sure we don't miss
            # incoming result files)
            n_reports_requested = n_trfs_requested = jobs_processed = n_pending = 0
            while not self._shutdown:
                try:
                    job = self._request_queue.get(block=False)
                except Empty:
                    if n_reports_requested or n_trfs_requested:
                        self.logger.info("%i requests processed, added %i reports and %i TRFs to queue", jobs_processed, n_reports_requested, n_trfs_requested)
                        if n_pending:
                            self.logger.warning("%i TRFs already pending", n_pending)
                    elif jobs_processed:
                        self.logger.info("%i requests processed, no new jobs", jobs_processed)
                    break
                jobs_processed += 1
                # check whether job already exists
                try:
                    if any(job.is_same(j) for j in self._user_jobs):
                        continue
                except Exception as error:
                    self.logger.error(f"Error processing new job:\nJob: {job}\nError: {error}")
                    continue

                # initialize job
                try:
                    with self.e_lock:
                        job.init_sub_jobs()
                        # check whether all target files already exist
                        if not job.test_path and not job.missing_trfs and not job.has_followup_jobs():
                            continue
                except Exception:
                    n_exceptions += 1
                    if n_exceptions == 1:
                        self.logger.error("Error initializing job %r", job)
                        print_traceback(sys.exc_info())
                    continue

                # file the job for secondary tasks once TRFs are done
                self._user_jobs.append(job)
                if job.test_path:
                    # file report job, using the test-path to uniquely identify it
                    self._report_jobs[job.test_path] = job
                    if not exists(job.test_path):
                        n_reports_requested += job.report

                # if all TRFs are already available
                if not job.trf_jobs:
                    self._job_finalize_queue.append(job)

                # schedule all TRF requests
                for trfjob in job.trf_jobs:
                    if trfjob.path in self._trf_jobs:
                        problems = assert_trf_jobs_equal(trfjob, self._trf_jobs[trfjob.path])
                        if problems:
                            self.logger.warning("Mismatching jobs for %s:\n%s", trfjob.path, '\n'.join(problems))
                    else:
                        self._trf_jobs[trfjob.path] = trfjob
                        if self.server.job_exists(trfjob.path):
                            n_pending += 1
                        elif job.priority:
                            self._trf_job_queue.appendleft(trfjob.path)
                        else:
                            self._trf_job_queue.append(trfjob.path)
                        n_trfs_requested += 1

            if n_exceptions:
                self.logger.error(f"Ignored {n_exceptions} faulty jobs")
                n_exceptions = 0

            # put a new TRF-job into the server queue
            if self._trf_job_queue and not self._shutdown and not self.server.full():
                path = self._trf_job_queue.popleft()
                func = trfjob = None
                if path not in self._trf_jobs:
                    self.logger.error("Trying to queue non-existing job: %s", path)
                else:
                    trfjob = self._trf_jobs[path]
                    try:
                        with self.e_lock:
                            func = trfjob.generate_job()
                    except Exception:
                        n_trf_exceptions += 1
                        if n_trf_exceptions == 1:
                            self.logger.error("Error processing trf-job: %s", path)
                            print_traceback(sys.exc_info())
                        # remove jobs related to the failed TRF
                        del self._trf_jobs[path]
                        for job in reversed(self._user_jobs):
                            if path in job.missing_trfs:
                                self._user_jobs.remove(job)
                                if job.test_path in self._report_jobs:
                                    del self._report_jobs[job.test_path]
                if func is not None:
                    try:
                        self.server.put(path, func)
                    except JobServerTerminated:
                        self.logger.info("Request rejected, server terminated")
                    except Exception:
                        self.logger.exception("Request rejected by server: %s", trfjob.desc)
                    else:
                        self.logger.info("Request %s", trfjob.desc)
            elif n_trf_exceptions:
                self.logger.error(f"Ignored {n_trf_exceptions} faulty trf-jobs")
                n_trf_exceptions = 0

            # receive all available results
            while True:
                try:
                    trf_path = self.server.get(block=False)
                except Empty:
                    break
                else:
                    if trf_path in self._trf_jobs:
                        trfjob = self._trf_jobs.pop(trf_path)
                        self.logger.info("Received  %s", trfjob.desc)
                    else:
                        self.logger.info("Received orphan  %s", trf_path)

                    for job in self._user_jobs:
                        if job.missing_trfs:
                            job.missing_trfs.discard(trf_path)
                            if not job.missing_trfs:
                                self._job_finalize_queue.append(job)

            # finalize jobs which received all TRFs
            while self._job_finalize_queue:
                job = self._job_finalize_queue.popleft()
                if job.test_path:
                    if job.report and not exists(job.test_path):
                        self._report_queue.put(job)

                if job.has_followup_jobs():
                    with self.e_lock:
                        for new_job in job.get_followup_jobs(self.logger):
                            self._request_queue.put(new_job)
                elif self._notifier:
                    message = fmtxt.FMText([f"All TRFs received for {job.name}.", fmtxt.linebreak])
                    message.append(self.show_jobs())
                    if not self._request_queue.empty():
                        message.append(fmtxt.linebreak)
                        message.append(fmtxt.linebreak)
                        message.append("Processing new requests...")
                    self._notifier.send(f'Job done: {job.name}', message)

            if self.server.terminated:
                return

            # make sure we don't keep checking empty queues
            if self.server.full() or not self._trf_job_queue:
                cycle_time = time() - cycle_start_time
                if cycle_time < MIN_IO_CYCLE_TIME:
                    sleep(MIN_IO_CYCLE_TIME - cycle_time)

    def _find_jobs(self, model: str):
        return [job for job in self._user_jobs if fnmatch.fnmatch(job.name, model)]

    def cancel_job(self, model: str):
        """Cancel one or several jobs

        Parameters
        ----------
        model
            Pattern to match jobs by model name. For example, a job's full
            model, or '*' to match all jobs.
        """
        jobs = self._find_jobs(model)
        if not jobs:
            raise ValueError(f"{model=}: no job with this model name")
        n_removed = 0
        for job in jobs:
            if job.trf_jobs is None:
                continue  # (peviously canceled)
            for trf_job in job.trf_jobs:
                if trf_job.path in self._trf_jobs:
                    del self._trf_jobs[trf_job.path]
                    if trf_job.path in self._trf_job_queue:
                        self._trf_job_queue.remove(trf_job.path)
                    n_removed += 1
            job.cancel()
        print(f"{len(jobs)} jobs with {n_removed} TRF-jobs canceled")

    def clear_report_queue(self):
        "Remove all report requests (but leave TRF-requests)"
        while True:
            try:
                job = self._report_queue.get(False)
                if job.test_path in self._report_jobs:
                    del self._report_jobs[job.test_path]
                    job.test_path = False
            except Empty:
                break

    def flush(self):
        """Remove finished jobs from list

        See Also
        --------
        clear_report_queue : skip queued report
        """
        for job in reversed(self._user_jobs):
            if job.missing_trfs:
                continue
            if job.test_path and not exists(job.test_path):
                continue
            if job.test_path:
                del self._report_jobs[job.test_path]
            self._user_jobs.remove(job)

    def info(self):
        out = fmtxt.Report(f"{self.server.host} ({self.server.ip}:{self.server.port})", date='%c')
        out.append(self.show_workers())
        out.append(fmtxt.linebreak)
        out.append(self.show_jobs(True))
        print(out)

    def make_reports(self, block=False, notify=False):
        "Make reports with calling thread"
        if notify and not self._notifier:
            raise ValueError("Can't notify because no notifier is available. Set the notify parameter when initializing the Dispatcher.")
        if block:
            print("Make all incoming reports; ctrl-c to stop...")

        n_made = 0
        while True:
            try:
                job = self._report_queue.get(block, 1000)
            except Empty:
                if not block:
                    print("Report queue empty")
                    if notify and n_made:
                        self._notifier.send("All queued reports are done.", f"{n_made} reports created.")
                    return
            except KeyboardInterrupt:
                break
            else:
                with self.e_lock:
                    job.make_test_report()
                n_made += 1
                sleep(2)  # make lock available

    def prioritize(self, model: str = None, priority: bool = True):
        """Set the priority of jobs with name matching ``model``

        Currently this only affects scheduling of new TRFs, i.e. a change only
        takes effect when a model is reduced.
        """
        priority = bool(priority)
        jobs = self._find_jobs(model)
        for job in jobs:
            job.priority = priority

    def _requeue_failed_jobs(self):
        while True:
            key = self.server.get_failed(True)
            if key is None:
                break
            elif key in self._trf_jobs:
                self._trf_job_queue.appendleft(key)

    def remove_broken_worker(self, worker, blacklist=False):
        """Move jobs sent to this worker back into the queue"""
        keys = [str(job.path) for job in self.server.remove_broken_worker(worker, blacklist)]
        if not keys:
            print(f"No jobs found for worker {worker}")
            return
        n_added = 0
        n_skipped = 0
        for key in keys:
            if key in self._trf_jobs:
                self._trf_job_queue.appendleft(key)
                n_added += 1
            else:
                n_skipped += 1
        msg = f"{n_added} jobs added back into queue"
        if n_skipped:
            msg += f"; {n_skipped} unknown jobs skipped"
        print(msg)

    def remove_broken_jobs(self, pattern):
        """Re-queue jobs for which processing failed

        Parameters
        ----------
        pattern : int | str | list
            Job model or comparison, job path pattern, or one or more job IDs.

        Notes
        -----
        Move jobs back into the queue based on target filename pattern. Assumes
        that the corresponding jobs are not being worked on anymore. Otherwise
        they will be received as orphans and overwrite
        """
        # check if pattern is a model
        if isinstance(pattern, str):
            model_jobs = [job for job in self._user_jobs if job.trf_jobs and fnmatch.fnmatch(job.name, pattern)]
        else:
            model_jobs = None
        # find TRF-job keys
        if model_jobs:
            keys = {trfjob.path for job in model_jobs for trfjob in job.trf_jobs}
            keys.intersection_update(job.path for job in self.server.pending_jobs())
            keys = list(keys)
        else:
            keys = self.server.find_jobs(pattern, 'pending')

        if not keys:
            print("No jobs match pattern")
            return
        prefix = commonprefix(keys)
        t = fmtxt.Table('llll')
        t.cells("Job", "Worker", "Orphan", "File")
        t.midrule()
        i_prefix = keys[0].rfind('/', 0, len(prefix)) + 1
        t.caption('\n'.join((
            f"Common prefix: {keys[0][:i_prefix]}",
            f"Orphan: Marked jobs are unknown (cannot be requeued)",
            f"File: Target file already exists",
        )))
        for key in keys:
            desc = key[i_prefix:]
            if key in self._trf_jobs:
                orphan = '\u2610'
                file_desc = '\u2611' if exists(key) else '\u2610'
            else:
                orphan = '\u2612'
                file_desc = ''
            t.cells(desc, self.server._jobs[key].worker, orphan, file_desc)
        print(t)
        command = ask(f"Remove {len(keys)} jobs?", {'requeue': 'requeue jobs', 'drop': 'drop jobs', 'abort': "don't do anything (default)"}, allow_empty=True)
        if command in ('requeue', 'drop'):
            n_skipped = n_restarted = 0
            self.server.remove_broken_job(keys)
            for key in keys:
                if key in self._trf_jobs:
                    if command == 'requeue':
                        self._trf_job_queue.appendleft(key)
                        n_restarted += 1
                    else:
                        pass  # FIXME: remove job properly
                else:
                    n_skipped += 1
            print(f"{n_restarted} restarted, {n_skipped} skipped")

    def show_jobs(self, trfs=False):
        pending_jobs = {job.path: job for job in self.server.pending_jobs()}
        priority = len({job.priority for job in self._user_jobs}) > 1
        t = fmtxt.Table('lllrrl' + 'l' * priority)
        t.cells("Exp.", "Epoch", "Model", "TRFs", "Pending")
        if priority:
            t.cell("Priority")
        t.cell('Report')
        t.midrule()
        for job in self._user_jobs:
            if job.trf_jobs is None:
                n_trfs = '<not'
                n_missing = 'initialized>'
                report = ''
            else:
                n_trfs = len(job.trf_jobs)
                n_missing = len(job.missing_trfs)
                if job.test_path:
                    if exists(job.test_path):
                        report = '\u2611'
                    else:
                        report = '\u2610'
                elif job.test_path is False:
                    report = '\u2612'
                else:
                    report = ''
            if isinstance(job, FuncIterJob):
                t.cells('<iter>', '')
            else:
                t.cell(job.experiment.__class__.__name__)  # Exp
                t.cell(job.options.get('epoch', ''))  # Epoch
            t.cell(job.name)
            t.cell(n_trfs)  # TRFs
            t.cell(n_missing)  # Pending
            if priority:
                t.cell(job.priority)
            t.cell(report)
            # TRFs currently being processed
            if trfs and job.trf_jobs:
                trf_jobs = [j for j in job.trf_jobs if j.path in pending_jobs]
                n = Counter(pending_jobs[j.path].worker or 'requested' for j in trf_jobs)
                for worker in sorted(n):
                    t.cells('', '')
                    t.cell(self.server._worker_info.get(worker, worker), just='r')
                    t.cells('', n[worker])
                    t.endline()
        return t

    def shutdown(self, block=True):
        "Schedule regular shutdown, waiting for outstanding results"
        if not self._shutdown:
            self.logger.info("Initiating shutdown...")
            self._shutdown = True
            # empty queue
            try:
                while True:
                    self._request_queue.get(block=False)
            except Empty:
                pass
            self.server.shutdown(True)  # need to join, otherwise will hang
        if block:
            self.join()

    def join(self):
        if not self._shutdown:
            raise RuntimeError("Can not join before shutting down")
        self.server.join()
        self._thread.join()
        self._requeue_thread.join()
        self.terminated = True
        self.logger.info("Dispatcher successfully terminated.")

    def __del__(self):
        if not self.terminated:
            self.shutdown(True)
