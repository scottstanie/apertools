"""
This module exports a Log class that wraps the logging python package

Uses the standard python logging utilities, just provides
nice formatting out of the box.

Usage:

    from apertools.log import get_log
    logger = get_log()

    logger.info("Something happened")
    logger.warning("Something concerning happened")
    logger.error("Something bad happened")
    logger.critical("Something just awful happened")
    logger.debug("Extra printing we often don't need to see.")
    # Custom output for this module:
    logger.success("Something great happened: highlight this success")
"""
import logging
import time

try:
    from colorlog import ColoredFormatter
    COLORS = True
except ImportError:
    from logging import Formatter
    COLORS = False


def get_log(debug=False, name=__file__, verbose=False):
    """Creates a nice log format for use across multiple files.

    Default logging level is INFO

    Args:
        name (Optional[str]): The name the logger will use when printing statements
        debug (Optional[bool]): If true, sets logging level to DEBUG

    """
    logger = logging.getLogger(name)
    return format_log(logger, debug=debug, verbose=verbose)


def format_log(logger, debug=False, verbose=False):
    """Makes the logging output pretty and colored with times"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'cyan',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'black,bg_red',
        'SUCCESS': 'white,bg_blue'
    }

    if COLORS:
        format_ = '[%(asctime)s] [%(log_color)s%(levelname)s %(filename)s%(reset)s] %(message)s%(reset)s'
        formatter = ColoredFormatter(format_, datefmt='%m/%d %H:%M:%S', log_colors=log_colors)
    else:
        format_ = '[%(asctime)s] [%(levelname)s %(filename)s] %(message)s'
        formatter = Formatter(format_, datefmt='%m/%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.SUCCESS = 25  # between WARNING and INFO
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')
    setattr(logger, 'success', lambda message, *args: logger._log(logging.SUCCESS, message, args))

    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(log_level)

        if verbose:
            logger.info('Logger initialized: %s' % (logger.name, ))

    if debug:
        logger.setLevel(debug)

    return logger


logger = get_log()


def log_runtime(f):
    """
    Logs how long a decorated function takes to run

    Args:
        f (function): The function to wrap

    Returns:
        function: The wrapped function

    Example:
        >>> @log_runtime
        ... def test_func():
        ...    return 2 + 4
        >>> test_func()
        6

    This prints out to stderr the following in addition to the answer:
    [05/26 10:05:51] [INFO log.py] Total elapsed time for test_func (minutes): 0.00

    """

    def wrapper(*args, **kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_time = t2 - t1
        time_string = 'Total elapsed time for {} : {} minutes ({} seconds)'.format(
            f.__name__,
            "{0:.2f}".format(elapsed_time / 60.0),
            "{0:.2f}".format(elapsed_time),
        )

        logger.info(time_string)
        return result

    return wrapper
