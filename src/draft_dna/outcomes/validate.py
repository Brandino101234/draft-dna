"""Validate the candidate value definitions against awards and contracts.

Everything is compared at the same point: N seasons after the draft, for classes
that have been observed for at least N seasons. Targets are things the value metric
never sees directly:

- All-NBA / All-Star selections through N seasons (voters' judgment)
- Second-contract pay: mean salary as % of cap in seasons 5-7 (teams' judgment),
  compared with value through season 4 (what teams knew when they paid)
- Peak pay: max salary as % of cap in seasons 5..N

Confidence intervals come from a bootstrap over players (resample players with
replacement 1,000 times, recompute every metric).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table
from draft_dna.outcomes.value import CANDIDATES

CAREER_N = 10
EARLY_N = 4
N_BOOT = 1000


def salary_pct_cap(s: Settings) -> pd.DataFrame:
    sal = read_table("staging", "bbref", "player_salaries", s)
    cap = read_table("staging", "bbref", "salary_cap", s).set_index("season")["salary_cap"]
    per = sal.groupby(["bbref_id", "season"], as_index=False)["salary"].sum()  # traded mid-season
    per["pct_cap"] = per["salary"] / per["season"].map(cap)
    return per


def validation_frame(s: Settings, otn: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per drafted player observed >= CAREER_N seasons, with values and targets."""
    drafted = players[players["drafted"]].set_index("bbref_id")
    at_n = otn[otn["n"] == CAREER_N].set_index("bbref_id")
    at_early = otn[otn["n"] == EARLY_N].set_index("bbref_id")
    ids = at_n.index.intersection(drafted.index)
    df = pd.DataFrame(index=ids)
    df["draft_year"] = drafted.loc[ids, "draft_year"]
    df["pick"] = drafted.loc[ids, "pick_overall"]
    for cand in CANDIDATES:
        df[f"career_{cand}"] = at_n.loc[ids, f"composite_{cand}"]
        df[f"early_{cand}"] = at_early.loc[ids, f"composite_{cand}"]
        df[f"peakonly_{cand}"] = at_n.loc[ids, f"peak3_{cand}"]
        df[f"totalonly_{cand}"] = at_n.loc[ids, f"total_{cand}"]
    df["all_star_n"] = at_n.loc[ids, "all_star_selections"]
    df["all_nba_n"] = at_n.loc[ids, "all_nba_selections"]

    pay = salary_pct_cap(s)
    pay = pay.join(drafted["draft_year"], on="bbref_id", how="inner")
    pay["n"] = pay["season"] - pay["draft_year"]
    second = pay[pay["n"].between(5, 7)].groupby("bbref_id")["pct_cap"].sum() / 3
    peak = pay[pay["n"].between(5, CAREER_N)].groupby("bbref_id")["pct_cap"].max()
    # No NBA salary in those seasons = paid nothing by the league (out of the NBA).
    df["second_contract_pct_cap"] = second.reindex(ids).fillna(0.0)
    df["peak_pay_pct_cap"] = peak.reindex(ids).fillna(0.0)
    return df


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    return float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else np.nan


def _rho(y: np.ndarray, x: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


# (target column, score prefix, metric, label)
CHECKS: list[tuple[str, str, Callable[[np.ndarray, np.ndarray], float], str]] = [
    ("all_nba_n", "career", lambda y, x: _auc(y > 0, x), "AUC: ever All-NBA (10 yrs)"),
    ("all_star_n", "career", lambda y, x: _auc(y > 0, x), "AUC: ever All-Star (10 yrs)"),
    ("all_star_n", "career", _rho, "Spearman: All-Star selections"),
    ("peak_pay_pct_cap", "career", _rho, "Spearman: peak pay % of cap (yrs 5-10)"),
    ("second_contract_pct_cap", "early", _rho, "Spearman: 2nd-contract pay vs value thru yr 4"),
]


def evaluate(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Point estimate and 95% bootstrap CI for every candidate x check, plus each
    candidate's difference from VORP (paired bootstrap: same resample for both)."""
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(df), len(df)) for _ in range(n_boot)]
    rows = []
    for target, prefix, fn, label in CHECKS:
        y = df[target].to_numpy()
        boots: dict[str, np.ndarray] = {}
        for cand in CANDIDATES:
            x = df[f"{prefix}_{cand}"].to_numpy()
            boots[cand] = np.array([fn(y[i], x[i]) for i in idx])
            rows.append(
                {
                    "check": label,
                    "candidate": cand,
                    "estimate": fn(y, x),
                    "ci_low": np.nanpercentile(boots[cand], 2.5),
                    "ci_high": np.nanpercentile(boots[cand], 97.5),
                }
            )
        for cand in CANDIDATES[1:]:
            diff = boots[cand] - boots["vorp"]
            rows[-len(CANDIDATES) + CANDIDATES.index(cand)].update(
                {
                    "diff_vs_vorp": float(np.nanmean(diff)),
                    "diff_ci_low": np.nanpercentile(diff, 2.5),
                    "diff_ci_high": np.nanpercentile(diff, 97.5),
                }
            )
    return pd.DataFrame(rows)


def composite_weight_check(df: pd.DataFrame, cand: str) -> pd.DataFrame:
    """Does combining peak and total beat either alone? (same checks, point estimates)"""
    rows = []
    for target, prefix, fn, label in CHECKS:
        if prefix != "career":
            continue
        y = df[target].to_numpy()
        for name, col in (
            ("peak only", f"peakonly_{cand}"),
            ("total only", f"totalonly_{cand}"),
            ("50/50 composite", f"career_{cand}"),
        ):
            rows.append({"check": label, "definition": name, "estimate": fn(y, df[col].to_numpy())})
    return pd.DataFrame(rows)
