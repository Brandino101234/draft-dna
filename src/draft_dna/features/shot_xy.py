"""Shot DNA, tier B: where a player shoots from (x/y coordinates).

Coordinates are standardized to feet from the basket for both leagues:
- ESPN college: x in feet across the court (0-50, center 25); the basket sits at
  y = 1.0 (fit from data: it best separates threes from twos in every line era).
- stats.nba.com: LOC_X / LOC_Y are tenths of feet from the basket.

The court is folded left/right (|dx|): which side a player favors is mostly noise at
these sample sizes, and folding halves the number of cells to estimate.

Each eligible player's shots become a smoothed density map (a kernel density estimate:
every shot is spread over nearby cells with a Gaussian bump, so a few hundred shots
give a stable picture). Non-negative matrix factorization then finds a few "style"
maps such that every player's map is approximately a non-negative mix of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF

from draft_dna.features.era import nba_line_ft, ncaa_line_ft

ESPN_BASKET_Y = 1.0
GRID_X = np.arange(0, 26)  # |dx| in feet, 1-ft cells
GRID_Y = np.arange(-4, 31)  # dy in feet (behind the basket to 30 ft out)
BANDWIDTH_FT = 1.5
MIN_XY_FGA = 100  # eligibility for spatial features (audit, D025)
MIN_XY_COVERAGE = 0.4
INCONSISTENT_FT = 2.0
CORNER_DY_FT = 8.0
CORNER_DX_FT = 20.0


def standardize_espn(shots: pd.DataFrame) -> pd.DataFrame:
    """Feet from the basket; coordinates contradicting the shot's value are dropped."""
    out = shots.copy()
    out["dx"] = out["x"] - 25.0
    out["dy"] = out["y"] - ESPN_BASKET_Y
    out["is_three"] = out["score_value"].eq(3) | out["shot_type"].str.contains("three", case=False)
    out["line_ft"] = out["season"].map(ncaa_line_ft)
    return _consistency_filter(out)


def standardize_nba(shots: pd.DataFrame) -> pd.DataFrame:
    out = shots.copy()
    out["dx"] = out["LOC_X"] / 10.0
    out["dy"] = out["LOC_Y"] / 10.0
    out["is_three"] = out["SHOT_TYPE"].str.startswith("3PT")
    out["made"] = out["SHOT_MADE_FLAG"].astype(bool)
    out["line_ft"] = out["season"].map(nba_line_ft)
    return _consistency_filter(out, corner_line_ft=22.0)


def _consistency_filter(df: pd.DataFrame, corner_line_ft: float | None = None) -> pd.DataFrame:
    dist = np.hypot(df["dx"], df["dy"])
    line = (
        df["line_ft"]
        if corner_line_ft is None
        else np.where(np.abs(df["dx"]) >= corner_line_ft - 0.5, corner_line_ft, df["line_ft"])
    )
    bad = (df["is_three"] & (dist < line - INCONSISTENT_FT)) | (
        ~df["is_three"] & (dist > line + INCONSISTENT_FT)
    )
    out = df.copy()
    out["dist_ft"] = dist
    out.loc[bad, ["dx", "dy", "dist_ft"]] = np.nan
    out["xy_valid"] = out["dx"].notna()
    return out


def zone(df: pd.DataFrame) -> pd.Series:
    """rim (<4 ft) / short mid (4-14) / long mid (14+ two) / corner three / above-break three."""
    d = df["dist_ft"]
    z = pd.Series(pd.NA, index=df.index, dtype="string")
    two = ~df["is_three"]
    z[two & (d < 4)] = "rim"
    z[two & (d >= 4) & (d < 14)] = "short_mid"
    z[two & (d >= 14)] = "long_mid"
    corner = df["is_three"] & (df["dx"].abs() >= CORNER_DX_FT) & (df["dy"] <= CORNER_DY_FT)
    z[corner] = "corner_three"
    z[df["is_three"] & ~corner & d.notna()] = "above_break_three"
    return z


def density_map(dx: np.ndarray, dy: np.ndarray, bandwidth: float = BANDWIDTH_FT) -> np.ndarray:
    """Smoothed shot density on the folded half-court grid, summing to 1."""
    hist, _, _ = np.histogram2d(
        np.abs(dx),
        dy,
        bins=[np.append(GRID_X, GRID_X[-1] + 1) - 0.5, np.append(GRID_Y, GRID_Y[-1] + 1) - 0.5],
    )
    smooth = gaussian_filter(hist, sigma=bandwidth, mode="constant")
    total = smooth.sum()
    return (smooth / total if total > 0 else smooth).ravel()


def player_maps(shots: pd.DataFrame, key: str = "bbref_id") -> pd.DataFrame:
    """One row per player: flattened density map (columns c0..cN) and shot count."""
    valid = shots[shots["xy_valid"]]
    rows = {}
    for pid, g in valid.groupby(key):
        rows[pid] = density_map(g["dx"].to_numpy(), g["dy"].to_numpy())
    maps = pd.DataFrame.from_dict(rows, orient="index")
    maps.columns = [f"c{i}" for i in range(maps.shape[1])]
    maps["xy_fga"] = valid.groupby(key).size()
    return maps


def fit_styles(maps: pd.DataFrame, k: int = 6, seed: int = 0) -> NMF:
    x = maps.filter(regex=r"^c\d+$").to_numpy()
    return NMF(n_components=k, init="nndsvda", max_iter=1000, random_state=seed).fit(x)


def style_weights(model: NMF, maps: pd.DataFrame) -> pd.DataFrame:
    """Each player's mix of styles, normalized to sum to 1."""
    w = model.transform(maps.filter(regex=r"^c\d+$").to_numpy())
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return pd.DataFrame(w, index=maps.index, columns=[f"style_{i}" for i in range(w.shape[1])])


def component_grid(model: NMF, i: int) -> np.ndarray:
    return model.components_[i].reshape(len(GRID_X), len(GRID_Y))
