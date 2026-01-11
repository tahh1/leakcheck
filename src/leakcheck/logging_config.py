import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Configure basic logging for leakcheck."""
    logging.basicConfig(level=level, format=log_format)
