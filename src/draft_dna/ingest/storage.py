"""Read and write layer tables as parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from draft_dna.config import Settings, get_settings
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

Layer = Literal["raw", "staging", "modeled"]


def table_path(layer: Layer, source: str, name: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    base = {"raw": s.paths.raw / "tables", "staging": s.paths.staging, "modeled": s.paths.modeled}
    return base[layer] / source / f"{name}.parquet"


def write_table(
    df: pd.DataFrame, layer: Layer, source: str, name: str, settings: Settings | None = None
) -> Path:
    path = table_path(layer, source, name, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s/%s/%s: %d rows", layer, source, name, len(df))
    return path


def read_table(
    layer: Layer, source: str, name: str, settings: Settings | None = None
) -> pd.DataFrame:
    return pd.read_parquet(table_path(layer, source, name, settings))
