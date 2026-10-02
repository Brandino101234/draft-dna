"""Phase 6: who beats their projection, and why.

Building blocks:
- As-of-draft projected ranges at N = 4, 6, 8 seasons (model of record: draft-slot
  history + conformal; class c trains year Y only if c + N <= Y).
- PIT score: where the actual outcome landed inside the projected range (0-1). If
  projections are calibrated, PIT is uniform; a group whose PIT averages above 0.5
  systematically beats its projections.
- Early-career situation for every drafted player (team quality, positional competition,
  coaching changes, trades, games absent), all measured from data, none hand-coded.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import backtest as bt
from draft_dna.eval import metrics as M
from draft_dna.eval.phase3 import pick_conformal
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

HORIZONS = (4, 6, 8)
FRANCHISE = {"SEA": "OKC", "NJN": "BRK", "VAN": "MEM", "CHH": "CHO", "CHA": "CHO",
             "NOH": "NOP", "NOK": "NOP", "WSB": "WAS"}  # fmt: skip
NBA_POS = {"PG": "guard", "SG": "guard", "SF": "forward", "PF": "forward", "C": "big"}


def pit(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability integral transform with mid-point handling of point masses
    (a player at exactly 0 whose projection puts 30% mass at 0 gets PIT ~0.15)."""
    upper = M.cdf_at_each(q, y)
    lower = M.cdf_at_each(q, y - 1e-9)
    return (lower + upper) / 2


def asof_bands(s: Settings) -> pd.DataFrame:
    """Projected quantiles at N = 4, 6, 8 for every drafted player with a projection,
    plus actual outcomes, PIT, and band verdicts (below floor / within / beat ceiling)."""
    df = bt.modeling_frame(s)
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    out = df[
        ["bbref_id", "player_name", "draft_year", "pick", "position", "prospect_source"]
    ].copy()
    for n in HORIZONS:
        col = f"y{n}"
        df[col] = df["bbref_id"].map(otn[otn["n"] == n].set_index("bbref_id")["peak3_blend"])
        known = df[df[col].notna()]
        rows = []
        for year in sorted(df["draft_year"].unique().astype(int)):
            train = known[known["draft_year"] + n <= year]
            cls = df[(df["draft_year"] == year) & df[col].notna()]
            if len(train) < 100 or cls.empty:
                continue
            model = pick_conformal().fit(train, train[col].to_numpy())
            q = M.monotone(model.predict_quantiles(cls))
            y = cls[col].to_numpy()
            rows.append(pd.DataFrame({
                "bbref_id": cls["bbref_id"].to_numpy(),
                f"floor{n}": q[:, M.qidx(M.FLOOR)], f"median{n}": q[:, M.qidx(M.MEDIAN)],
                f"ceiling{n}": q[:, M.qidx(M.CEILING)], f"actual{n}": y, f"pit{n}": pit(q, y),
            }))  # fmt: skip
        band = pd.concat(rows, ignore_index=True)
        band[f"verdict{n}"] = np.select(
            [band[f"actual{n}"] > band[f"ceiling{n}"], band[f"actual{n}"] < band[f"floor{n}"]],
            ["beat ceiling", "below floor"], "within band",
        )  # fmt: skip
        out = out.merge(band, on="bbref_id", how="left")
    return out


def situations(s: Settings) -> pd.DataFrame:
    """Early-career context for each drafted player (Y = draft year, seasons Y+1..Y+3)."""
    players = read_table("modeled", "core", "players", s)
    picks = players[players["drafted"]][["bbref_id", "draft_year", "team_id"]].copy()
    picks["draft_year"] = picks["draft_year"].astype(int)
    feats = read_table("modeled", "features", "predraft", s).set_index("bbref_id")
    picks["position"] = picks["bbref_id"].map(feats["position"])
    picks["franchise"] = picks["team_id"].replace(FRANCHISE)

    teams = read_table("staging", "bbref", "nba_team_seasons", s)
    tq = teams.set_index(["season", "team"])[["srs", "wins", "losses"]]
    pre = tq.reindex(list(zip(picks["draft_year"], picks["team_id"], strict=True)))
    picks["team_srs_pre"] = pre["srs"].to_numpy()
    # Bottom third of the league by SRS in the season just before the draft.
    rank = teams.groupby("season")["srs"].rank(pct=True)
    weak = teams.assign(pct=rank).set_index(["season", "team"])["pct"]
    picks["weak_team"] = (
        weak.reindex(list(zip(picks["draft_year"], picks["team_id"], strict=True))).to_numpy()
        <= 1 / 3
    )

    # Positional competition: share of the drafting team's pre-draft-season minutes played
    # at the prospect's position (guard / forward / big).
    st = read_table("staging", "bbref", "nba_team_stints", s)
    st = st[st["phase"] == "regular"].assign(
        pos_bucket=lambda d: d["pos"].str.split("-").str[0].map(NBA_POS)
    )
    team_min = st.groupby(["season", "team"])["mp"].sum()
    pos_min = st.groupby(["season", "team", "pos_bucket"])["mp"].sum()
    keys = list(zip(picks["draft_year"], picks["team_id"], picks["position"], strict=True))
    share = [pos_min.get(k, np.nan) / team_min.get(k[:2], np.nan) for k in keys]
    picks["position_minutes_share"] = share
    med = pd.Series(share).groupby(picks["position"].to_numpy()).transform("median").to_numpy()
    picks["crowded_position"] = picks["position_minutes_share"].to_numpy() >= med

    # Coaching instability on the drafting team during the rookie deal.
    coaches = read_table("staging", "bbref", "nba_coaches", s)
    head = coaches.sort_values("games_coached").groupby(["season", "team"]).last()["coach_id"]
    changes = []
    for y, t in zip(picks["draft_year"], picks["team_id"], strict=True):
        seq = [head.get((y + k, t)) for k in range(0, 4)]
        seq = [c for c in seq if c is not None]
        changes.append(sum(a != b for a, b in pairwise(seq)))
    picks["coach_changes_y1_3"] = changes
    picks["coach_change"] = picks["coach_changes_y1_3"] >= 1

    # Player-level early career (only meaningful for players who reached the NBA).
    seasons = read_table("modeled", "core", "nba_player_seasons", s)
    early = seasons[seasons["season_num"].between(1, 3)]
    st_early = st.merge(picks[["bbref_id", "draft_year", "team_id"]], on="bbref_id")
    st_early = st_early[(st_early["season"] - st_early["draft_year"]).between(1, 3)]
    other_team = st_early[st_early["team"] != st_early["team_id"]].groupby("bbref_id").size()
    picks["played_y1_3"] = picks["bbref_id"].isin(early["bbref_id"])
    picks["traded_early"] = picks["bbref_id"].isin(other_team.index)
    y12 = (
        seasons[seasons["season_num"].between(1, 2)]
        .groupby("bbref_id")[["games_absent", "season_games"]]
        .sum()
    )
    picks["absent_share_y1_2"] = picks["bbref_id"].map(y12["games_absent"] / y12["season_games"])
    # Missed >= 25% of games in years 1-2. Mostly role (G League, DNP), not only injury:
    # there is no reliable public injury history (D005), so this is "low availability".
    picks["low_availability"] = picks["absent_share_y1_2"] >= 0.25
    y1 = seasons[seasons["season_num"] == 1].set_index("bbref_id")
    picks["rookie_minutes"] = picks["bbref_id"].map(y1["mp"]).fillna(0.0)
    return picks


def run(s: Settings) -> pd.DataFrame:
    bands = asof_bands(s)
    sit = situations(s)
    out = bands.merge(sit.drop(columns=["draft_year", "position"]), on="bbref_id", how="left")
    write_table(out, "modeled", "phase6", "player_outcomes_vs_projection", s)
    log.info("phase 6 table: %d players; with year-6 PIT: %d", len(out), out["pit6"].notna().sum())
    return out
