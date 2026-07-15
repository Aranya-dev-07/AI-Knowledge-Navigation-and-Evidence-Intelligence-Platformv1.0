"""Centralized logging configuration for TrustLens AI backend.

Configures the root Python logger once, at application startup, with
a Rich-powered console handler (or a plain structured formatter for
non-interactive/production environments) based on ``core.config``.
Every other module should obtain its logger via
``logging.getLogger(__name__)`` (or the ``get_logger`` helper exposed
here) after ``setup_logging`` has run in ``main.py``.
"""

import logging
import sys

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from core.config import settings

_LOG_RECORD_FORMAT: str = "%(name)s: %(message)s"
_PLAIN_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are useful at DEBUG for the application but
# excessively verbose for general use; capped at WARNING regardless of
# the application's own configured log level.
_NOISY_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "uvicorn.access",
    "sqlalchemy.engine.Engine",
    "asyncio",
)

_configured: bool = False


def setup_logging() -> None:
    """Configure the root logger for the entire application process.

    Installs either a Rich-formatted console handler (``LOG_FORMAT ==
    "rich"``, ideal for local development) or a plain, structured
    handler suitable for log aggregation systems (``LOG_FORMAT ==
    "json"``-equivalent plain text, safe for production log
    collectors). Idempotent: calling this function more than once
    (e.g. under test reloads) has no additional effect after the
    first successful configuration.

    Returns:
        None
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)

    # Remove any pre-existing handlers (e.g. Uvicorn's default handlers)
    # to avoid duplicate log lines once our handler is attached.
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)

    if settings.LOG_FORMAT == "rich":
        install_rich_traceback(show_locals=settings.DEBUG)
        handler: logging.Handler = RichHandler(
            level=settings.LOG_LEVEL,
            rich_tracebacks=True,
            tracebacks_show_locals=settings.DEBUG,
            markup=False,
            show_path=settings.DEBUG,
        )
        handler.setFormatter(logging.Formatter(_LOG_RECORD_FORMAT, datefmt=_DATE_FORMAT))
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_PLAIN_LOG_FORMAT, datefmt=_DATE_FORMAT))

    handler.setLevel(settings.LOG_LEVEL)
    root_logger.addHandler(handler)

    for noisy_logger_name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    _configured = True

    root_logger.debug(
        "Logging configured: level=%s format=%s environment=%s",
        settings.LOG_LEVEL,
        settings.LOG_FORMAT,
        settings.ENVIRONMENT,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging has been configured.

    Convenience wrapper around ``logging.getLogger`` for modules that
    prefer an explicit accessor. Functionally equivalent to calling
    ``logging.getLogger(__name__)`` directly once ``setup_logging``
    has already run in ``main.py``.

    Args:
        name: The logger name, conventionally the caller's ``__name__``.

    Returns:
        logging.Logger: A logger instance scoped to ``name``.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)