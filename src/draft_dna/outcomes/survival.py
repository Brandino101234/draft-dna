"""Career length with censoring (survival analysis).

Duration = seasons from the draft through the player's last NBA season (0 = never
played). A career has *ended* (event) if the player has not appeared in either of the
last two completed seasons; otherwise it is *censored*: still going, length so far is a
lower bound. Two seasons of slack keep a player who missed one year injured from being
counted as retired.

- Kaplan-Meier: the share of a group still in the league after k seasons, using
  censored careers correctly (they count while observed, then drop out of the risk set).
- Cox proportional hazards: how pick, age and era shift the yearly risk of a career
  ending. A hazard ratio of 1.3 means a 30% higher chance each season of the career
  ending, all else equal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table

GRACE_SEASONS = 2
PICK_BANDS = [(1, 5, "Picks 1-5"), (6, 14, "Picks 6-14"), (15, 30, "Picks 15-30"),
              (31, 60, "Picks 31-60")]  # fmt: skip


def pick_band(pick: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=pick.index, dtype="string")
    for lo, hi, label in PICK_BANDS:
        out[(pick >= lo) & (pick <= hi)] = label
    return out


def career_table(s: Settings) -> pd.DataFrame:
    players = read_table("modeled", "core", "players", s)
    seasons = read_table("modeled", "core", "nba_player_seasons", s)
    last_complete = s.current_nba_season - 1
    p = players[players["drafted"] & (players["draft_year"] < last_complete)].copy()
    last = seasons.groupby("bbref_id")["season"].max()
    count = seasons.groupby("bbref_id")["season"].nunique()
    p["last_season"] = p["bbref_id"].map(last)
    p["seasons_played"] = p["bbref_id"].map(count).fillna(0).astype(int)
    p["duration"] = (p["last_season"] - p["draft_year"]).fillna(0).astype(int)
    p["ended"] = p["last_season"].isna() | (p["last_season"] <= last_complete - GRACE_SEASONS)
    draft_day = pd.to_datetime(p["draft_year"].astype(int).astype(str) + "-06-25")
    p["age_at_draft"] = (draft_day - p["birth_date"]).dt.days / 365.25
    p["pick_band"] = pick_band(p["pick_overall"])
    return p


def kaplan_meier(careers: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Survival curve with 95% CI per pick band, at season 0..horizon."""
    rows = []
    for _, _, band in PICK_BANDS:
        g = careers[careers["pick_band"] == band]
        km = KaplanMeierFitter().fit(g["duration"], event_observed=g["ended"], label=band)
        grid = np.arange(0, horizon + 1)
        surv = km.survival_function_at_times(grid).to_numpy()
        ci = km.confidence_interval_
        lo = np.interp(grid, ci.index, ci.iloc[:, 0])
        hi = np.interp(grid, ci.index, ci.iloc[:, 1])
        # "Still in the league after k seasons" = survived past duration k.
        for k, sv, low, high in zip(grid, surv, lo, hi, strict=True):
            rows.append(
                {"band": band, "seasons": int(k), "survival": sv, "ci_low": low,
                 "ci_high": high, "n": len(g), "median": km.median_survival_time_}
            )  # fmt: skip
    return pd.DataFrame(rows)


def cox(careers: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Hazard ratios for pick (log scale), age at draft, era and pre-draft background."""
    d = careers.dropna(subset=["age_at_draft"]).copy()
    d = d[d["duration"] > 0]  # never-played players have no NBA career to end
    d["log2_pick"] = np.log2(d["pick_overall"])
    d["age_at_draft"] = d["age_at_draft"] - 20
    d["draft_year_10"] = (d["draft_year"] - 2010) / 10
    d["non_college"] = (d["prospect_source"] != "college").astype(int)
    cols = ["log2_pick", "age_at_draft", "draft_year_10", "non_college"]
    cph = CoxPHFitter(penalizer=0.01).fit(
        d[[*cols, "duration", "ended"]], duration_col="duration", event_col="ended"
    )
    summ = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    summ.columns = ["hazard_ratio", "ci_low", "ci_high", "p"]
    return summ.reset_index(names="covariate"), float(cph.concordance_index_)
