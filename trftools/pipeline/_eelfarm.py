import logging


def start_dispatcher(notify=False, debug=False):
    """A dispatcher provides an experiment level interface to Eelfarm

    Parameters
    ----------
    notify : str
        Email addresss to notify when jobs are done.
    debug : bool
        Log debug messages from HTML server.

    Returns
    -------
    dispatcher : Dispatcher
        The dispatcher.
    """
    from eelfarm._utils import screen_handler
    from ._dispatcher import Dispatcher

    d = Dispatcher(notify=notify)
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
