"""Sports-Reference College Basketball: player seasons and team-season context.

Player pages are found via the college link on each Basketball-Reference bio
(`cbb_id`), so no searching is needed. Advanced player stats thin out before
~2010 (usage, AST%, REB% missing for the oldest seasons); we keep what exists.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from draft_dna.ingest.fetcher import Fetcher, NotFoundError
from draft_dna.ingest.html_tables import parse_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

SOURCE = "sports_reference"
BASE = "https://www.sports-reference.com/cbb"
PLAYER_TABLES = {
    "players_totals": "totals",
    "players_advanced": "advanced",
    "players_per_poss": "per_poss",
}
SEASON_TABLES = {
    "school-stats": "basic_school_stats",
    "advanced-school-stats": "adv_school_stats",
}


def player_url(cbb_id: str) -> str:
    return f"{BASE}/players/{cbb_id}.html"


def ingest_players(fetcher: Fetcher, cbb_ids: Iterable[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {name: [] for name in PLAYER_TABLES.values()}
    ids = sorted(set(cbb_ids))
    for i, cid in enumerate(ids, 1):
        if i % 100 == 0 or not fetcher.is_cached(player_url(cid)):
            log.info("cbb player pages: %d/%d (%s)", i, len(ids), cid)
        try:
            html = fetcher.get(player_url(cid))
        except NotFoundError:
            log.warning("no cbb page for %s", cid)
            continue
        for tid, name in PLAYER_TABLES.items():
            df = parse_table(html, tid)
            if df.empty:
                continue
            df = df[df["year_id"].str.match(r"\d{4}-\d{2}", na=False)]  # drop career rows
            df.insert(0, "cbb_id", cid)
            out[name].append(df)
    return {f"player_{k}": pd.concat(v, ignore_index=True) for k, v in out.items() if v}


def ingest_team_seasons(fetcher: Fetcher, seasons: Iterable[int]) -> dict[str, pd.DataFrame]:
    """All D-I teams per season: record, SRS, SOS, totals, and (when available) pace."""
    out: dict[str, list[pd.DataFrame]] = {k: [] for k in SEASON_TABLES}
    for s in seasons:
        for page, tid in SEASON_TABLES.items():
            try:
                html = fetcher.get(f"{BASE}/seasons/men/{s}-{page}.html")
            except NotFoundError:
                log.warning("no %s page for %d", page, s)
                continue
            df = parse_table(html, tid)
            df = df[df.get("school_name", pd.Series(dtype=str)).str.len() > 0]
            df.insert(0, "season", s)
            out[page].append(df)
    return {
        f"team_{k.replace('-', '_')}": pd.concat(v, ignore_index=True) for k, v in out.items() if v
    }
