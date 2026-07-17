"""! @file utils.py
@brief Shared utilities: logging setup, CSV I/O, root bracketing helpers.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LOGGER_NAME = "neutronstar"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """!Return the package logger, configuring a console handler on first use.

    @param name  Logger name (child loggers propagate to the package root).
    @return      A configured `logging.Logger`.
    """
    logger = logging.getLogger(name)
    root = logging.getLogger(LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logger


def write_csv(path: Path | str, header: Sequence[str],
              columns: Sequence[np.ndarray]) -> Path:
    """!Write named columns to a CSV file.

    @param path     Output file path (parent directories are created).
    @param header   Column names.
    @param columns  Sequence of equal-length 1-D arrays.
    @return         The resolved output path.
    @throws ValueError if the column lengths differ from each other.
    """
    cols = [np.asarray(c) for c in columns]
    n = {c.size for c in cols}
    if len(n) != 1:
        raise ValueError(f"CSV columns have inconsistent lengths: {sorted(n)}")
    if len(header) != len(cols):
        raise ValueError("header and columns must have the same length")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(zip(*[c.tolist() for c in cols], strict=True))
    return path


def write_kv_csv(path: Path | str, rows: Sequence[Mapping[str, object]]) -> Path:
    """!Write a list of dictionaries (same keys) to CSV.

    @param path  Output file path.
    @param rows  Sequence of mappings; keys of the first row define columns.
    @return      The resolved output path.
    @throws ValueError if `rows` is empty.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path
