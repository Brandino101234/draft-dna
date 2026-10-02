"""Export the small modeled tables the deployed app needs into app/bundle/.

The public app (Streamlit Community Cloud) has no database and no raw data. It reads this
bundle instead: grades, comps, style tables and the inputs for drawing cards on demand.
Only modeled outputs and the few columns cards display are exported; no raw source
tables. `draft-dna refresh` re-exports it, and committing app/bundle updates the live app.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from draft_dna.config import REPO_ROOT, Settings
from draft_dna.ingest.storage import read_table, table_path
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

BUNDLE_DIR = REPO_ROOT / "app" / "bundle"
# (source, name, columns to keep or None for all)
TABLES: list[tuple[str, str, list[str] | None]] = [
    ("grading", "grades", None),
    ("grading", "plays_like", None),
    ("grading", "style_map", None),
    ("grading", "trajectory_bands", None),
    ("grading", "nba_maps", None),
    ("grading", "rookie_tracker", None),
    ("projections", "comps", None),
    ("recruits", "players", None),
    ("recruits", "summary", None),
    ("recruits", "tests", None),
    ("features", "shot_maps", None),
    (
        "core",
        "players",
        [
            "bbref_id",
            "player_name",
            "draft_year",
            "pick_overall",
            "team_id",
            "drafted",
            "college_name",
            "pre_draft_org",
        ],
    ),
    ("features", "predraft", ["bbref_id", "age_at_draft", "height_in", "position"]),
    (
        "outcomes",
        "outcomes_through_n",
        [
            "bbref_id",
            "n",
            "peak3_graded",
            "all_star_selections",
            "all_nba_selections",
            "all_defense_selections",
            "dpoy_awards",
        ],
    ),
]


def export(s: Settings, out: Path = BUNDLE_DIR) -> Path:
    total = 0
    for source, name, cols in TABLES:
        df = read_table("modeled", source, name, s)
        if cols is not None:
            df = df[[c for c in cols if c in df.columns]]
        if name.endswith("maps"):  # density grids: 32-bit is plenty for drawing
            grid = df.columns[df.columns.str.fullmatch(r"c\d+")]
            df[grid] = df[grid].astype("float32")
        path = out / source / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        total += path.stat().st_size
    params = table_path("modeled", "outcomes", "params", s).with_suffix(".json")
    shutil.copy(params, out / "outcomes" / "params.json")
    log.info("app bundle: %d tables, %.1f MB in %s", len(TABLES), total / 1e6, out)
    return out
