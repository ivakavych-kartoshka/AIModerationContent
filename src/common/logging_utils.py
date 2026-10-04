"""Console + file logging shared by every entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "sensitiveai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, path: str | Path, level: int = logging.DEBUG) -> logging.Logger:
    """Also write DEBUG-level records to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    return logger


def set_level(level: str | int) -> None:
    logging.getLogger("sensitiveai").setLevel(level)
