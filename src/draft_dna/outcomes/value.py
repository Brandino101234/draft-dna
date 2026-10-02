"""Season value metrics and same-point career outcomes.

Three candidate definitions of a season's value:

- **A: VORP.** Value over replacement player (box plus-minus scaled by minutes).
  Simple and widely understood.
- **B: Blend.** The mean of z-scored VORP and Win Shares, two independently built
  value stats. Averaging diversifies each one's blind spots.
- **C: Factor.** The first factor of VORP, Win Shares, minutes, shrunken BPM and
  starts share: the single signal those measures share, with data-chosen weights.

Every candidate uses the same career framework, computed *through N seasons since the
draft*. A season out of the league counts as zero, so comparisons at the same N are fair:

- total(N)      sum of season values in seasons 1..N
- peak3(N)      best 3 consecutive seasons within 1..N, averaged
- longevity(N)  seasons with >= 500 minutes within 1..N
- playoff(N)    sum of playoff VORP / Win Shares within 1..N
- composite(N)  0.75 * z(peak3) + 0.25 * z(total), z-scored against the training
                classes at the same N (weights chosen in DECISIONS D016)
- peak3_graded(N)  like peak3 of the blend, but each season also counts its playoff
                value (playoff VORP and Win Shares on the same z scale, no shift, so no
                playoffs = 0). Used for grades (D033); accolade floors are applied there.

The primary metric is the blend (D016); all three are kept for transparency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

CANDIDATES = ("vorp", "blend", "factor")
PRIMARY = "blend"
PEAK_WEIGHT = 0.75
ROTATION_MINUTES = 500
BPM_PRIOR_MINUTES = 500  # shrink BPM toward replacement (-2.0) on small minutes
REPLACEMENT_BPM = -2.0
FACTOR_INPUTS = ["vorp", "ws", "mp", "bpm_shrunk", "start_share"]


def last_complete_season(s: Settings) -> int:
    return s.current_nba_season - 1


def season_values(
    seasons: pd.DataFrame, fit_mask: pd.Series
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add value_vorp / value_blend / value_factor to player-seasons.

    Standardization and the factor model are fit only on rows in `fit_mask` (training
    classes), then applied to everyone.
    """
    df = seasons.copy()
    df["bpm_shrunk"] = (
        df["bpm"].fillna(REPLACEMENT_BPM) * df["mp"] + REPLACEMENT_BPM * BPM_PRIOR_MINUTES
    ) / (df["mp"] + BPM_PRIOR_MINUTES)
    df["start_share"] = (df["games_started"].fillna(0) / df["games"].where(df["games"] > 0)).fillna(
        0
    )

    fit = df[fit_mask]
    df["value_vorp"] = df["vorp"]

    z = {}
    for col in ("vorp", "ws"):
        mu, sd = fit[col].mean(), fit[col].std()
        z[col] = (mu, sd)
        df[f"z_{col}"] = (df[col] - mu) / sd
    df["value_blend"] = (df["z_vorp"] + df["z_ws"]) / 2

    scaler = StandardScaler().fit(fit[FACTOR_INPUTS])
    fa = FactorAnalysis(n_components=1, random_state=0).fit(scaler.transform(fit[FACTOR_INPUTS]))
    loadings = pd.Series(fa.components_[0], index=FACTOR_INPUTS)
    sign = 1.0 if loadings["vorp"] >= 0 else -1.0  # orient so higher = better
    df["value_factor"] = sign * fa.transform(scaler.transform(df[FACTOR_INPUTS]))[:, 0]

    # Express blend and factor on a VORP-like scale where a season out of the league
    # (zero minutes) is 0: subtract the value a 0-minute season would get.
    zero = pd.DataFrame([{c: 0.0 for c in FACTOR_INPUTS}])
    zero["bpm_shrunk"] = REPLACEMENT_BPM
    zero_blend = ((0 - z["vorp"][0]) / z["vorp"][1] + (0 - z["ws"][0]) / z["ws"][1]) / 2
    zero_factor = sign * fa.transform(scaler.transform(zero[FACTOR_INPUTS]))[0, 0]
    df["value_blend"] -= zero_blend
    # Playoff value on the regular-season blend scale (playoff stats are counting stats,
    # so a deep run adds more). No shift: missing the playoffs adds 0.
    po = (df["po_vorp"].fillna(0) / z["vorp"][1] + df["po_ws"].fillna(0) / z["ws"][1]) / 2
    df["value_playoff"] = po
    df["value_factor"] -= zero_factor
    meta = {"factor_loadings": (sign * loadings).round(3).to_dict(), "z": z}
    return df, meta


def season_grid(players: pd.DataFrame, seasons: pd.DataFrame, last_season: int) -> pd.DataFrame:
    """One row per player x season number 1..N_observed, zero-filled when not in the NBA."""
    p = players[["bbref_id", "draft_year", "drafted", "first_season"]].copy()
    p["base_year"] = p["draft_year"].where(p["drafted"], p["first_season"] - 1)
    p = p.dropna(subset=["base_year"])
    p["n_observed"] = (last_season - p["base_year"]).clip(lower=0).astype(int)
    rows = p.loc[p.index.repeat(p["n_observed"])].copy()
    rows["season_num"] = rows.groupby("bbref_id").cumcount() + 1
    rows["season"] = (rows["base_year"] + rows["season_num"]).astype(int)
    value_cols = [c for c in seasons.columns if c.startswith("value_")] + [
        "mp", "games", "games_started", "vorp", "ws", "po_vorp", "po_ws", "po_mp", "all_star",
        "all_nba_team", "all_defense_team", "won_dpoy",
        "mvp_share", "season_num",
    ]  # fmt: skip
    s = seasons.copy()
    for col in ("all_defense_team", "won_dpoy"):  # optional award columns
        if col not in s:
            s[col] = pd.NA
    s = s[["bbref_id", "season", *[c for c in value_cols if c != "season_num"]]]
    grid = rows.merge(s, on=["bbref_id", "season"], how="left")
    grid["in_nba"] = grid["mp"].notna()
    counts = ["mp", "games", "games_started", "vorp", "ws", "po_vorp", "po_ws", "po_mp"]
    num = [c for c in grid.columns if c.startswith("value_")] + counts
    grid[num] = grid[num].fillna(0.0)
    grid["all_star"] = grid["all_star"].astype("boolean").fillna(False).astype(bool)
    grid["all_nba"] = grid["all_nba_team"].notna()
    grid["all_defense"] = grid["all_defense_team"].notna()
    grid["dpoy"] = grid["won_dpoy"].astype("boolean").fillna(False).astype(bool)
    grid["mvp_share"] = grid["mvp_share"].fillna(0.0)
    return grid.drop(columns=["all_nba_team", "all_defense_team", "won_dpoy"])


def _rolling_peak(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Running max of the trailing `window`-season average (shorter windows early on)."""
    out = np.empty(len(values))
    best = -np.inf
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        best = max(best, values[lo : i + 1].sum() / window)
        out[i] = best
    return out


def outcomes_through_n(grid: pd.DataFrame) -> pd.DataFrame:
    """Cumulative outcomes for every player at every N they have been observed."""
    g = grid.sort_values(["bbref_id", "season_num"]).copy()
    by = g.groupby("bbref_id", sort=False)
    out = g[["bbref_id", "season_num", "season", "in_nba"]].rename(columns={"season_num": "n"})
    for cand in CANDIDATES:
        col = f"value_{cand}"
        out[f"total_{cand}"] = by[col].cumsum()
        out[f"peak3_{cand}"] = by[col].transform(
            lambda v: pd.Series(_rolling_peak(v.to_numpy()), index=v.index)
        )
    playoff = g.get("value_playoff", 0.0)
    graded = g["value_blend"] + playoff
    out["peak3_graded"] = graded.groupby(g["bbref_id"], sort=False).transform(
        lambda v: pd.Series(_rolling_peak(v.to_numpy()), index=v.index)
    )
    out["minutes"] = by["mp"].cumsum()
    # Role anchors for tiers: best 3-season stretch of minutes and starts (per season).
    for src, dst in (("mp", "best3_minutes"), ("games_started", "best3_starts")):
        out[dst] = by[src].transform(
            lambda v: pd.Series(_rolling_peak(v.to_numpy()), index=v.index)
        )
    out["seasons_in_nba"] = by["in_nba"].cumsum()
    out["longevity"] = by["mp"].transform(lambda m: (m >= ROTATION_MINUTES).cumsum())
    out["playoff_vorp"] = by["po_vorp"].cumsum()
    out["playoff_ws"] = by["po_ws"].cumsum()
    out["playoff_minutes"] = by["po_mp"].cumsum()
    out["all_star_selections"] = by["all_star"].cumsum()
    out["all_nba_selections"] = by["all_nba"].cumsum()
    out["all_defense_selections"] = by["all_defense"].cumsum()
    out["dpoy_awards"] = by["dpoy"].cumsum()
    out["mvp_shares"] = by["mvp_share"].cumsum()
    return out.reset_index(drop=True)


def add_composites(otn: pd.DataFrame, training_ids: set[str]) -> pd.DataFrame:
    """composite_<cand>(N) = weighted z(peak3) and z(total), z-scored at the same N
    against training-class players only (so later classes don't move the scale)."""
    out = otn.copy()
    ref = out[out["bbref_id"].isin(training_ids)]
    for cand in CANDIDATES:
        parts = []
        for metric in (f"peak3_{cand}", f"total_{cand}"):
            stats = ref.groupby("n")[metric].agg(["mean", "std"])
            mu = out["n"].map(stats["mean"])
            sd = out["n"].map(stats["std"]).replace(0, np.nan)
            parts.append((out[metric] - mu) / sd)
        out[f"composite_{cand}"] = PEAK_WEIGHT * parts[0] + (1 - PEAK_WEIGHT) * parts[1]
    return out


def build(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    players = read_table("modeled", "core", "players", s)
    seasons = read_table("modeled", "core", "nba_player_seasons", s)
    training = players.loc[players["draft_class_role"] == "training", "bbref_id"]
    fit_mask = seasons["bbref_id"].isin(set(training))
    seasons, meta = season_values(seasons, fit_mask)
    grid = season_grid(players, seasons, last_complete_season(s))
    otn = add_composites(outcomes_through_n(grid), set(training))
    log.info("factor loadings: %s", meta["factor_loadings"])
    return seasons, otn, meta


def run(s: Settings) -> pd.DataFrame:
    seasons, otn, _ = build(s)
    keep = ["bbref_id", "season", *[c for c in seasons.columns if c.startswith("value_")],
            "bpm_shrunk", "start_share"]  # fmt: skip
    write_table(seasons[keep], "modeled", "outcomes", "season_values", s)
    write_table(otn, "modeled", "outcomes", "outcomes_through_n", s)
    return otn
