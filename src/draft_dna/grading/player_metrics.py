"""Extra per-player metrics for the app (D038). Descriptive, built after the fact: none of
these feed the projections.

- surplus_m: rookie-contract surplus. Each of a player's first four seasons is priced at
  what veterans with the same production were paid (share of the cap, fit on seasons 5+),
  minus what he was actually paid, summed and expressed in today's dollars.
- late_bloomer: how much his career peak grew from year 3 to year 8 beyond what players
  with the same year-3 peak typically gained (standardized residual; + = late bloomer).
- playoff_riser: playoff BPM minus regular-season BPM in the same seasons, weighted by
  playoff minutes and shrunk toward 0 for small samples (points per 100 possessions).
- second_pct / second_vs_slot: best salary as a share of the cap in seasons 5-7 after
  the draft, and that minus the average for his draft slot (classes with 7 seasons).
- rotation_seasons / availability: seasons with 1,000+ minutes, and the share of his
  team's games he played in seasons he was in the league.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

ROOKIE_SEASONS = 4
SECOND_CONTRACT = (5, 7)
RISER_PRIOR_MIN = 1000  # playoff minutes of shrinkage toward 0
MIN_PLAYOFF_MIN = 300
ROTATION_MIN = 1000


def _seasons(s: Settings) -> pd.DataFrame:
    ns = read_table("modeled", "core", "nba_player_seasons", s)
    vals = read_table("modeled", "outcomes", "season_values", s)
    sal = read_table("staging", "bbref", "player_salaries", s)
    cap = read_table("staging", "bbref", "salary_cap", s).set_index("season")["salary_cap"]
    d = ns.merge(vals[["bbref_id", "season", "value_blend"]], on=["bbref_id", "season"], how="left")
    pay = sal.groupby(["bbref_id", "season"])["salary"].sum().rename("salary").reset_index()
    d = d.merge(pay, on=["bbref_id", "season"], how="left")
    d["pct_cap"] = d["salary"] / d["season"].map(cap)
    # Missing salaries are mostly two-way / 10-day players: use that season's near-minimum.
    floor = d.groupby("season")["pct_cap"].quantile(0.05)
    d["pct_cap_filled"] = d["pct_cap"].fillna(d["season"].map(floor))
    return d


def market_rate(d: pd.DataFrame) -> tuple[float, float]:
    """Share of cap paid per unit of season value, from veterans (season 5+)."""
    vet = d[(d["season_num"] >= 5) & d["pct_cap"].notna() & d["value_blend"].notna()]
    x = vet["value_blend"].clip(lower=0).to_numpy()
    b, a = np.polyfit(x, vet["pct_cap"].to_numpy(), 1)
    return float(a), float(b)


def rookie_surplus(d: pd.DataFrame, cap_now: float) -> pd.Series:
    a, b = market_rate(d)
    floor = float(d["pct_cap"].quantile(0.05))
    rk = d[d["season_num"].between(1, ROOKIE_SEASONS)].copy()
    rk["market"] = np.maximum(a + b * rk["value_blend"].clip(lower=0), floor)
    rk["surplus"] = rk["market"] - rk["pct_cap_filled"]
    return rk.groupby("bbref_id")["surplus"].sum() * cap_now / 1e6


def late_bloomer(peaks: pd.DataFrame, nba_early: pd.Series) -> pd.Series:
    """Standardized residual of sqrt(peak year 8) on sqrt(peak year 3), for players who were
    in the NBA for 2+ of their first 3 seasons (draft-and-stash players would otherwise
    look like late bloomers just for arriving late)."""
    eligible = nba_early.index[nba_early >= 2]
    p = peaks.loc[peaks.index.intersection(eligible), [3, 8]].dropna().clip(lower=0) ** 0.5
    b, a = np.polyfit(p[3], p[8], 1)
    resid = p[8] - (a + b * p[3])
    return resid / resid.std()


def playoff_riser(d: pd.DataFrame) -> pd.Series:
    po = d[(d["po_mp"] > 0) & d["po_bpm"].notna() & d["bpm"].notna()]
    w = po["po_mp"]
    diff = (po["po_bpm"] - po["bpm"]) * w
    g = pd.DataFrame({"diff": diff, "w": w, "id": po["bbref_id"]}).groupby("id")
    raw = g["diff"].sum() / g["w"].sum()
    mins = g["w"].sum()
    out = raw * mins / (mins + RISER_PRIOR_MIN)
    return out[mins >= MIN_PLAYOFF_MIN]


def second_contract(d: pd.DataFrame) -> pd.Series:
    lo, hi = SECOND_CONTRACT
    sc = d[d["season_num"].between(lo, hi)]
    return sc.groupby("bbref_id")["pct_cap"].max()


def durability(d: pd.DataFrame) -> pd.DataFrame:
    played = d[d["games"] > 0]
    g = played.groupby("bbref_id")
    return pd.DataFrame(
        {
            "rotation_seasons": g["mp"].apply(lambda m: int((m >= ROTATION_MIN).sum())),
            "availability": g["games"].sum() / g["season_games"].sum(),
        }
    )


def build(s: Settings) -> pd.DataFrame:
    grades = read_table("modeled", "grading", "grades", s).set_index("bbref_id")
    cap = read_table("staging", "bbref", "salary_cap", s).set_index("season")["salary_cap"]
    cap_now = float(cap.get(s.current_nba_season, cap.iloc[-1]))
    d = _seasons(s)
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    from draft_dna.outcomes.tiers import graded_peak, tier_cuts

    otn["peak"] = graded_peak(otn, tier_cuts(s))
    peaks = otn.pivot(index="bbref_id", columns="n", values="peak")

    out = pd.DataFrame(index=grades.index)
    last_complete = s.current_nba_season - 1
    years_in = last_complete - grades["draft_year"]
    surplus = rookie_surplus(d, cap_now).reindex(out.index)
    # Out of the league for his rookie years = no surplus, not unknown.
    out["surplus_m"] = surplus.where(years_in < ROOKIE_SEASONS, surplus.fillna(0.0))
    early = d[d["season_num"].between(1, 3) & (d["games"] > 0)].groupby("bbref_id").size()
    out["late_bloomer"] = late_bloomer(peaks.reindex(out.index), early)
    out["playoff_riser"] = playoff_riser(d).reindex(out.index)
    second = second_contract(d).reindex(out.index)
    done = years_in >= SECOND_CONTRACT[1]
    out["second_pct"] = second.fillna(0.0).where(done)
    slot = (
        out.assign(pick=grades["pick"])
        .loc[done]
        .groupby("pick")["second_pct"]
        .mean()
        .rolling(5, center=True, min_periods=1)
        .mean()
    )
    out["second_vs_slot"] = out["second_pct"] - grades["pick"].map(slot)
    out = out.join(durability(d))
    out["rotation_seasons"] = out["rotation_seasons"].fillna(0).astype(int)
    # Where he came from, and when he arrived (draft-and-stash shows up as a late debut).
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    out["college"] = out.index.map(players["college_name"])
    out["pre_draft_team"] = out.index.map(players["pre_draft_org"])
    country = out["pre_draft_team"].str.extract(r"\(([^()]+)\)\s*$", expand=False)
    out["country"] = country.where(~country.str.fullmatch(r"[A-Z]{2}", na=False))  # US states
    played = d[d["games"] > 0]
    out["debut_season_num"] = played.groupby("bbref_id")["season_num"].min().reindex(out.index)
    out["ever_played"] = out["debut_season_num"].notna()
    log.info(
        "player metrics: surplus %d, late bloomer %d, playoff riser %d, second contract %d",
        out["surplus_m"].notna().sum(),
        out["late_bloomer"].notna().sum(),
        out["playoff_riser"].notna().sum(),
        out["second_pct"].notna().sum(),
    )
    return out.reset_index(names="bbref_id")


def run(s: Settings) -> None:
    write_table(build(s), "modeled", "grading", "player_metrics", s)
