"""Metrics that compare players and draft classes *across* years (D037).

A redraft ranks players within one class, so the #3 player of a stacked class and of a
weak one look the same. These put everyone on one historical scale:

- typical_curve: the typical (median) and average career peak at each pick, from classes
  whose careers are complete. Neighboring picks are pooled (wider for late picks, which are
  noisier) and the curve is forced to never rise with pick number.
- equivalent_pick: "played like a #X pick", the earliest pick whose typical career
  he matched. Isaiah Thomas (#60) played like a typical #2.
- all_time_rank: rank of a player's peak among every pick since 1996.
- class_strength: total peak value a class produced minus what an average class produces
  from the same picks (positive = stronger draft), plus stars and depth. Unfinished
  careers count at their *expected* (mean) peak, which includes the chance of stardom:
  scoring them at the median would make every unfinished class look weak, because
  careers are skewed by a few superstars.

Peaks are the graded peaks (incl. playoffs and accolade floors, D033/D034); for careers
still in progress the current projected median is used, so those classes are provisional.
Light module (pandas/numpy/sklearn only): the deployed app imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REFERENCE_LAST_CLASS = 2017  # every career in these classes has 8+ seasons
MAX_PICK = 60
LOTTERY = 14
STAR_TIERS = ("All-Star", "All-NBA", "Superstar", "MVP", "Legend")
STARTER_TIERS = ("Starter", *STAR_TIERS)


def _window(pick: int) -> int:
    return round(pick * 0.2)  # #1-2 alone, +-1 for #3-7, ... +-12 at #60


def typical_curve(grades: pd.DataFrame, last_class: int = REFERENCE_LAST_CLASS) -> pd.DataFrame:
    """Median and mean career peak by pick (1..60) over complete classes, smoothed."""
    ref = grades[(grades["draft_year"] <= last_class) & grades["pick"].between(1, MAX_PICK)]
    picks = np.arange(1, MAX_PICK + 1)
    med, mean = [], []
    for p in picks:
        w = _window(int(p))
        v = ref.loc[ref["pick"].between(p - w, p + w), "current_median"]
        med.append(v.median())
        mean.append(v.mean())
    iso = IsotonicRegression(increasing=False)
    return pd.DataFrame(
        {
            "pick": picks,
            "typical_peak": iso.fit_transform(picks, med),
            "average_peak": IsotonicRegression(increasing=False).fit_transform(picks, mean),
        }
    )


def equivalent_pick(values: pd.Series, curve: pd.DataFrame) -> pd.Series:
    """Earliest pick whose typical career this value matches (1 = at least a typical #1).
    NaN when the value is at or below what a typical late pick produces (no NBA value)."""
    typical = curve["typical_peak"].to_numpy()
    picks = curve["pick"].to_numpy()
    floor = typical[-1]

    def one(v: float) -> float:
        if pd.isna(v) or v <= floor + 1e-9:
            return np.nan
        hit = np.nonzero(typical <= v)[0]
        return float(picks[hit[0]]) if len(hit) else float(MAX_PICK)

    return values.map(one)


def equivalent_pick_label(p: float) -> str:
    if pd.isna(p):
        return "no NBA value"
    return "#1 or better" if p <= 1 else f"#{int(p)}"


def expected_peak(grades: pd.DataFrame) -> pd.Series:
    """Mean of each player's current quantile grid (5th..95th pct, 5% apart) plus the two
    2.5% tails: the low tail at the 5th pct, the high tail extended by the last gap.
    Equals the actual peak once a career is finished (the grid collapses to it)."""
    q = grades[[c for c in grades.columns if c.startswith("post_q")]].to_numpy(dtype=float)
    tail_hi = q[:, -1] + (q[:, -1] - q[:, -2])
    return pd.Series(0.95 * q.mean(axis=1) + 0.025 * q[:, 0] + 0.025 * tail_hi, index=grades.index)


def all_time_rank(grades: pd.DataFrame) -> pd.Series:
    """Rank of each player's (current) peak among every graded pick, 1 = best."""
    return grades["current_median"].rank(ascending=False, method="min").astype(int)


def class_strength(grades: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """One row per draft class: value over an average class, stars, starters, best player."""
    d = grades[grades["pick"].between(1, MAX_PICK)].copy()
    d["expected"] = d["pick"].map(curve.set_index("pick")["average_peak"])
    d["over_average"] = expected_peak(d) - d["expected"]
    d["finished"] = d["status"].str.startswith("Career")
    d["lottery_over"] = d["over_average"].where(d["pick"] <= LOTTERY, 0.0)
    d["late_over"] = d["over_average"].where(d["pick"] > LOTTERY, 0.0)
    best = d.loc[d.groupby("draft_year")["current_median"].idxmax()].set_index("draft_year")
    out = d.groupby("draft_year").agg(
        picks=("bbref_id", "size"),
        strength=("over_average", "sum"),
        lottery=("lottery_over", "sum"),
        later_picks=("late_over", "sum"),
        stars=("current_tier", lambda t: int(t.isin(STAR_TIERS).sum())),
        starters_plus=("current_tier", lambda t: int(t.isin(STARTER_TIERS).sum())),
        top5_value=("current_median", lambda v: float(v.nlargest(5).sum())),
        finished_share=("finished", "mean"),
    )
    out["best_player"] = best["player_name"]
    out["provisional"] = out["finished_share"] < 0.9
    out["rank"] = out["strength"].rank(ascending=False, method="min").astype(int)
    return out.reset_index()


def pick_value(curve: pd.DataFrame) -> pd.DataFrame:
    """Draft pick value chart: each slot's average career peak, with the #1 pick = 100."""
    v = curve[["pick", "average_peak"]].copy()
    v["value"] = 100 * v["average_peak"] / v["average_peak"].iloc[0]
    return v
