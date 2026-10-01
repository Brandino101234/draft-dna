"""Load staging and modeled parquet tables into the DuckDB database.

Schemas mirror the layers: `staging.<source>__<table>` and `modeled.<area>__<table>`.
The .duckdb file is disposable; rebuild it with `draft-dna build`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from draft_dna.config import Settings, get_settings
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)


def connect(settings: Settings | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    s = settings or get_settings()
    s.paths.database.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(s.paths.database), read_only=read_only)


def _tables(layer_dir: Path) -> list[tuple[str, Path]]:
    return [(f"{p.parent.name}__{p.stem}", p) for p in sorted(layer_dir.glob("*/*.parquet"))]


def load(settings: Settings | None = None) -> list[str]:
    s = settings or get_settings()
    loaded = []
    with connect(s) as con:
        for schema, layer_dir in (("staging", s.paths.staging), ("modeled", s.paths.modeled)):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            for name, path in _tables(layer_dir):
                con.execute(
                    f"CREATE OR REPLACE TABLE {schema}.{name} AS "
                    f"SELECT * FROM read_parquet('{path.as_posix()}')"
                )
                loaded.append(f"{schema}.{name}")
    log.info("loaded %d tables into %s", len(loaded), s.paths.database)
    return loaded
