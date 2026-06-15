"""utils/logging_utils.py — Structured logging setup."""
from __future__ import annotations
import logging, sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers, force=True)
    for noisy in ("transformers", "datasets", "accelerate", "tokenizers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
