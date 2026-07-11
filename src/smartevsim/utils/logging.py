"""Application logging configuration."""

from logging.config import dictConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with the package's standard format.

    Args:
        level: Root logging level name, such as ``"INFO"`` or ``"DEBUG"``.
    """
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )
