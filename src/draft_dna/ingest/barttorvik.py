"""Barttorvik (T-Rank) college data, 2008 onward.

Player CSVs have no header row; the column order below is verified by tests
(rim + mid attempts == 2PA, and known players' picks and birthdates).
`year` is the season's ending year (2019 = 2018-19).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

import pandas as pd

from draft_dna.ingest.fetcher import Fetcher, NotFoundError
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

SOURCE = "barttorvik"
BASE = "https://barttorvik.com"
FIRST_SEASON = 2008

PLAYER_COLUMNS = [
    "player_name", "team", "conf", "gp", "min_pct", "ortg", "usg", "efg", "ts_pct",
    "orb_pct", "drb_pct", "ast_pct", "tov_pct", "ftm", "fta", "ft_pct", "two_pm",
    "two_pa", "two_pct", "three_pm", "three_pa", "three_pct", "blk_pct", "stl_pct",
    "ftr", "class_yr", "height", "jersey", "porpag", "adjoe", "pfr", "year", "pid",
    "hometown", "recruit_rating", "ast_tov", "rim_made", "rim_att", "mid_made",
    "mid_att", "rim_pct", "mid_pct", "dunks_made", "dunks_att", "dunk_pct", "nba_pick",
    "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "mpg", "ogbpm",
    "dgbpm", "oreb_pg", "dreb_pg", "treb_pg", "ast_pg", "stl_pg", "blk_pg", "pts_pg",
    "role", "three_pa_per100", "birth_date",
]  # fmt: skip


def parse_player_csv(body: bytes) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(body.decode("utf-8", errors="replace"))))
    bad = [r for r in rows if len(r) != len(PLAYER_COLUMNS)]
    if bad:
        raise ValueError(f"{len(bad)} rows with unexpected width (expected {len(PLAYER_COLUMNS)})")
    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def parse_team_csv(body: bytes) -> pd.DataFrame:
    text = body.decode("utf-8", errors="replace")
    # The final header cell is "Fun Rk, adjt" (two columns in one quoted cell).
    text = text.replace('"Fun Rk, adjt"', "fun_rk,adjt", 1)
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "") for c in df.columns]
    return df


def ingest_players(fetcher: Fetcher, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for y in seasons:
        body = fetcher.get(f"{BASE}/getadvstats.php", {"year": y, "csv": 1}, ext="csv")
        frames.append(parse_player_csv(body))
    return pd.concat(frames, ignore_index=True)


def ingest_teams(fetcher: Fetcher, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for y in seasons:
        try:
            body = fetcher.get(f"{BASE}/{y}_team_results.csv", ext="csv")
        except NotFoundError:
            log.warning("no Barttorvik team results for %d", y)
            continue
        df = parse_team_csv(body)
        df.insert(0, "year", y)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
