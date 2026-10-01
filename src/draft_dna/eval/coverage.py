"""Data coverage report by draft class: what share of picks have each data type.

Writes reports/phase1/coverage_by_draft_year.png and coverage_by_draft_year.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from draft_dna import db
from draft_dna.config import Settings, get_settings
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

# Sequential blue ramp (light -> dark), from the project's chart palette.
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

COVERAGE_SQL = """
WITH picks AS (
    SELECT * FROM modeled.core__players WHERE drafted
),
college AS (
    SELECT bbref_id,
           max((usg_pct IS NOT NULL)::INT) AS has_usage,
           max((bart_bpm IS NOT NULL)::INT) AS has_bpm,
           max((bart_rim_att IS NOT NULL)::INT) AS has_rim_mid
    FROM modeled.core__college_player_seasons GROUP BY 1
),
nba AS (SELECT DISTINCT bbref_id FROM modeled.core__nba_player_seasons),
comb AS (
    SELECT bbref_id,
           max((wingspan IS NOT NULL)::INT) AS has_measure,
           max((max_vertical_leap IS NOT NULL)::INT) AS has_athletic
    FROM modeled.core__combine GROUP BY 1
)
SELECT p.draft_year,
       count(*) AS picks,
       avg((p.prospect_source = 'college')::INT) AS "Pre-draft: college",
       avg((p.prospect_source = 'other_team')::INT) AS "Pre-draft: intl / pro team",
       avg((p.prospect_source = 'high_school')::INT) AS "Pre-draft: high school",
       avg((p.cbb_id IS NOT NULL)::INT) AS "College box stats",
       avg(coalesce(c.has_usage, 0)) AS "College usage / AST% / REB%",
       avg((p.bart_pid IS NOT NULL)::INT) AS "Barttorvik linked",
       avg(coalesce(c.has_bpm, 0)) AS "College BPM",
       avg(coalesce(c.has_rim_mid, 0)) AS "College rim / mid splits",
       avg(coalesce(m.has_measure, 0)) AS "Combine measurements",
       avg(coalesce(m.has_athletic, 0)) AS "Combine athletic tests",
       avg((n.bbref_id IS NOT NULL)::INT) AS "Played in NBA"
FROM picks p
LEFT JOIN college c USING (bbref_id)
LEFT JOIN nba n USING (bbref_id)
LEFT JOIN comb m USING (bbref_id)
GROUP BY 1 ORDER BY 1
"""


def coverage_table(s: Settings) -> pd.DataFrame:
    with db.connect(s, read_only=True) as con:
        return con.execute(COVERAGE_SQL).df()


def plot_coverage(df: pd.DataFrame, path: Path) -> None:
    fields = [c for c in df.columns if c not in ("draft_year", "picks")]
    data = df[fields].T.to_numpy(dtype=float) * 100
    years = df["draft_year"].astype(int).tolist()

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(15, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, len(years)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(fields)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v >= 55 else TEXT_PRIMARY)  # fmt: skip
    ax.set_xticks(range(len(years)), [str(y)[2:] for y in years], fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(fields)), fields, fontsize=10, color=TEXT_PRIMARY)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Draft class ('96 to '26)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(
        "Share of draft picks with each data type (%)",
        loc="left", fontsize=13, color=TEXT_PRIMARY, pad=12,
    )  # fmt: skip
    fig.text(
        0.01, 0.01,
        "Sources: Basketball-Reference, Sports-Reference CBB, Barttorvik, stats.nba.com. "
        "Pre-draft rows describe the player pool; other rows are data availability.",
        fontsize=8, color=TEXT_SECONDARY,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    show = df.copy()
    for c in show.columns:
        if c not in ("draft_year", "picks"):
            show[c] = (show[c] * 100).round(0).astype(int)
    path.write_text(
        "# Data coverage by draft class\n\n"
        "Percent of draft picks in each class with the data type available.\n\n"
        + show.to_markdown(index=False)
        + "\n"
    )


def run(s: Settings | None = None) -> pd.DataFrame:
    s = s or get_settings()
    df = coverage_table(s)
    out = s.paths.reports / "phase1"
    plot_coverage(df, out / "coverage_by_draft_year.png")
    write_markdown(df, out / "coverage_by_draft_year.md")
    log.info("coverage report written to %s", out)
    return df
