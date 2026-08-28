import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIRECTORY = Path(__file__).resolve().parent.parent / "logs"
APPLICATION_LOG = LOG_DIRECTORY / "application.log"
ERROR_LOG = LOG_DIRECTORY / "error.log"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

_configured = False


def setup_logging() -> None:
    global _configured

    if _configured:
        return

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    application_handler = RotatingFileHandler(
        APPLICATION_LOG,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    application_handler.setLevel(logging.INFO)
    application_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        ERROR_LOG,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(application_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
