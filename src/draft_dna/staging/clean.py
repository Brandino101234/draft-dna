"""Typed conversions for scraped string columns."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

_NUM_JUNK = re.compile(r"[,$%+]")


def to_num(s: pd.Series) -> pd.Series:
    """'1,234' / '$9,757,440' / '.442' / '+8.6' / '' -> float (NaN when empty)."""
    cleaned = s.astype("string").str.strip().str.replace(_NUM_JUNK, "", regex=True)
    return pd.to_numeric(cleaned.replace("", pd.NA), errors="coerce")


def numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = to_num(out[c])
    return out


def height_to_inches(s: pd.Series) -> pd.Series:
    """'6-9' -> 81.0"""
    parts = s.astype("string").str.extract(r"^(\d+)-(\d+)$")
    return pd.to_numeric(parts[0]) * 12 + pd.to_numeric(parts[1])


def season_end_year(s: pd.Series) -> pd.Series:
    """'2018-19' -> 2019; '1999-00' -> 2000."""
    start = pd.to_numeric(s.astype("string").str.slice(0, 4), errors="coerce")
    return start + 1


def blank_to_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({"": pd.NA})
