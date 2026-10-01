"""Era context tables for normalizing stats across seasons.

`nba_era`: league pace, scoring efficiency, 3PA rate and 3-point line distance by
season. `ncaa_era`: the same for Division I, computed from all-team totals, plus
the NCAA line distance (it moved twice: 2008-09 and 2019-20).

Possessions are estimated with the standard formula
    poss ~= FGA - ORB + TOV + 0.475 * FTA
which works on box-score totals and so covers seasons where published pace is missing.
"""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, table_path, write_table

# Season ending year -> 3-point line distance in feet (above the break).
NBA_LINE_FT = {1995: 22.0, 1996: 22.0, 1997: 22.0}  # shortened line; 23.75 otherwise
NBA_STANDARD_LINE_FT = 23.75
NBA_CORNER_FT = 22.0


def ncaa_line_ft(season: int) -> float:
    if season <= 2008:
        return 19.75
    if season <= 2019:
        return 20.75
    return 22.146  # 22' 1.75"


def nba_line_ft(season: int) -> float:
    return NBA_LINE_FT.get(season, NBA_STANDARD_LINE_FT)


def possessions(fga: pd.Series, orb: pd.Series, tov: pd.Series, fta: pd.Series) -> pd.Series:
    return fga - orb + tov + 0.475 * fta


def build_nba_era(s: Settings) -> pd.DataFrame:
    la = read_table("staging", "bbref", "nba_league_averages", s)
    la = la[la["season"] >= s.draft_classes.training[0] - 3].copy()
    la["three_line_ft"] = la["season"].map(nba_line_ft)
    la["corner_three_ft"] = NBA_CORNER_FT
    return la.rename(columns={"off_rtg": "ortg"}).sort_values("season").reset_index(drop=True)


def build_ncaa_era(s: Settings) -> pd.DataFrame:
    t = read_table("staging", "cbb", "team_seasons", s)
    t = t[t["g"] > 0]
    g = t.groupby("season")
    totals = g[["fga", "fg3a", "fta", "ft", "pts", "orb", "tov", "g", "mp"]].sum(min_count=1)
    out = pd.DataFrame(index=totals.index)
    out["n_teams"] = g.size()
    out["fg3a_rate"] = totals["fg3a"] / totals["fga"]
    out["ts_pct"] = totals["pts"] / (2 * (totals["fga"] + 0.44 * totals["fta"]))
    out["ft_rate"] = totals["fta"] / totals["fga"]
    out["pts_per_g"] = totals["pts"] / totals["g"]
    # ORB/TOV are missing for some early team-seasons; only estimate pace when complete.
    complete = t.dropna(subset=["fga", "orb", "tov", "fta", "g"])
    poss = possessions(complete["fga"], complete["orb"], complete["tov"], complete["fta"])
    est = (poss / complete["g"]).groupby(complete["season"]).mean()
    out["poss_per_g_est"] = est
    out["pace_coverage"] = complete.groupby("season").size() / out["n_teams"]
    if "team_pace" in t:
        out["pace_published"] = g["team_pace"].mean()
    out["ortg_est"] = 100 * totals["pts"] / (out["poss_per_g_est"] * totals["g"])
    out = out.reset_index()
    out["three_line_ft"] = out["season"].map(ncaa_line_ft)
    return out


def run(s: Settings) -> None:
    write_table(build_nba_era(s), "modeled", "era", "nba_era", s)
    if table_path("staging", "cbb", "team_seasons", s).exists():
        write_table(build_ncaa_era(s), "modeled", "era", "ncaa_era", s)
