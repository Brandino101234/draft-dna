"""Outcome tiers as cutoffs on peak value (best 3-season stretch of the blend metric).

Tiers are defined on the *same quantity the models predict*, so a predicted
distribution of peak value converts directly into tier probabilities.

Cutoffs are calibrated on training classes against observable anchors:
    Out of league  < 1,000 career minutes
    Bust           in the league, but never a rotation player
    Rotation       best 3-year stretch averaging >= 1,000 minutes
    Starter        best 3-year stretch averaging >= 41 starts and >= 1,800 minutes
    All-Star       at least one All-Star selection
    All-NBA        at least one All-NBA selection
Each cutoff is the value that best separates neighboring anchor groups (maximum
balanced accuracy), so the tiers mean "plays like a typical X", not "was voted X".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["Out of league", "Bust", "Rotation", "Starter", "All-Star", "All-NBA"]


def role_anchor(row: pd.Series) -> int:
    """Index into TIERS from awards and role (minutes/starts), not from value."""
    if row["all_nba_selections"] > 0:
        return 5
    if row["all_star_selections"] > 0:
        return 4
    if row["best3_starts"] >= 41 and row["best3_minutes"] >= 1800:
        return 3
    if row["best3_minutes"] >= 1000:
        return 2
    if row["minutes"] >= 1000:
        return 1
    return 0


def best_cut(lower: np.ndarray, upper: np.ndarray) -> float:
    """Cutoff maximizing balanced accuracy between two groups (lower < cut <= upper)."""
    cands = np.unique(np.concatenate([lower, upper]))
    best, best_ba = float(cands[0]), -1.0
    for c in cands:
        ba = ((lower < c).mean() + (upper >= c).mean()) / 2
        if ba > best_ba:
            best, best_ba = float(c), ba
    return best


def calibrate(peak: pd.Series, anchor: pd.Series) -> list[float]:
    """Five cutoffs between the six tiers, forced to be increasing."""
    cuts: list[float] = []
    for k in range(1, len(TIERS)):
        c = best_cut(peak[anchor == k - 1].to_numpy(), peak[anchor == k].to_numpy())
        cuts.append(max(c, cuts[-1] + 1e-6) if cuts else c)
    return [round(c, 2) for c in cuts]


def assign(peak: pd.Series | np.ndarray, cuts: list[float]) -> np.ndarray:
    """Tier index for each peak value."""
    return np.searchsorted(np.asarray(cuts), np.asarray(peak, dtype=float), side="right")
