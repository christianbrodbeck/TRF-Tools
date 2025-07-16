import logging

from ._dispatcher import Dispatcher


def start_dispatcher(
        notify: str = False,
        debug: bool = False,
        job_queue_length: int = 2,
        host: str = None,
        port: int = 8000,
):
    """A dispatcher provides an experiment level interface to Eelfarm

    Parameters
    ----------
    notify
        Email address to notify when jobs are done.
    debug
        Log debug messages from HTML server.
    job_queue_length
        Number of jobs that will be kept in memory in a queue while waiting for
        a worker. Pre-loading more jobs allows faster dispatching when workers
        request jobs, but also requires more memory.
        With the minimum (0) The dispatcher will only start loading the
        next job once the current job has been claimed by a worker (this can
        still lead to multiple jobs in memory, while the data is being
        transferred to the worker).
    host
        Manually specify the host name or IP address (in cases where
        :func:`socket.gethostbyname` fails with the default, which is
        :func:`socket.gethostname`).
    port
        Port to use for server.

    Returns
    -------
    Dispatcher
        The dispatcher.

    Examples
    --------
    Start a dispatcher and add some jobs::

        import trftools

        d = trftools.start_dispatcher(job_queue_length=1)
        d.add_jobs_from_file('project/pipline/jobs.py')

    Eelfarm workers with the host's IP will now start processing these jobs.
    Display information about job status::

        d.info()

    You can cancel jobs, using model patterns with asterisks. For example, to
    cancel all jobs that start with ``'gammatone + '``, use::

        d.cancel_jobs('gammatone + *')

    This will stop generating jobs (but if jobs were already sent to the eelfarm
    worker they cannot be stopped without killing the worker).
    To stop the dispatcher properly (without losing jobs)::

        d.shutdown()

    You can now quit the Python session. Jobs that were unfinished on workers
    at the time of shutdown will be delivered the next time the dispatcher is
    initialized again.
    """
    from eelfarm._utils import screen_handler

    d = Dispatcher(host, port, job_queue_length, notify)
    # configure logging
    d.logger.addHandler(screen_handler)
    d.logger.setLevel(logging.DEBUG)
    d.server.logger.addHandler(screen_handler)  # eelfarm.server
    d.server.logger.setLevel(logging.DEBUG)
    if not debug:
        logger = logging.getLogger('eelfarm.server.http')
        logger.setLevel(logging.INFO)
    d.start()
    return d
