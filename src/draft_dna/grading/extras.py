"""Supporting tables for cards and the app.

- trajectory_bands: draft-night projected range of peak value through each season 1..8,
  so a 2022-2025 player's path can be drawn against what was expected at each point.
- plays_like: NBA "plays like" style comps. NBA early-career shot maps are projected onto
  the same six NMF college styles (court distances rescaled so each league's 3-point line
  sits at the same radius). Nearest NBA players by style mix, within 3 inches of height.
  Stylistic only: it says nothing about how good a player will be.
- style_map: 2D t-SNE layout of every player's style mix (college and NBA early career).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from draft_dna.config import Settings
from draft_dna.eval import metrics as M
from draft_dna.features import shot_xy
from draft_dna.grading.run import QCOLS, asof_prior_grids
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

STYLE_K = 6
MIN_NBA_FGA = 200
N_PLAYS_LIKE = 5
MAX_HEIGHT_DIFF = 3.0


def trajectory_bands(s: Settings) -> pd.DataFrame:
    rows = []
    for n in range(1, 9):
        g = asof_prior_grids(s, horizon=n, retrospective=True)
        q = g[QCOLS].to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "bbref_id": g.index,
                    "n": n,
                    "floor": q[:, M.qidx(M.FLOOR)],
                    "median": q[:, M.qidx(M.MEDIAN)],
                    "ceiling": q[:, M.qidx(M.CEILING)],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def nba_maps(s: Settings) -> pd.DataFrame:
    shots = read_table("raw", "nba_api", "shots_early_career", s)
    std = shot_xy.standardize_nba(shots)
    counts = std[std["xy_valid"]].groupby("bbref_id").size()
    keep = counts.index[counts >= MIN_NBA_FGA]
    return shot_xy.player_maps(std[std["bbref_id"].isin(keep)])


def nba_zone_shares(s: Settings) -> pd.DataFrame:
    """Share of each NBA player's early-career shots by zone (same zones as college)."""
    shots = read_table("raw", "nba_api", "shots_early_career", s)
    std = shot_xy.standardize_nba(shots)
    std = std[std["xy_valid"]]
    counts = std.groupby("bbref_id").size()
    std = std[std["bbref_id"].isin(counts.index[counts >= MIN_NBA_FGA])].copy()
    std["zone"] = shot_xy.zone(std)
    z = std.groupby("bbref_id")["zone"].value_counts(normalize=True).unstack(fill_value=0)
    return z.add_prefix("zone_")


def readable_axes(zones: pd.DataFrame) -> pd.DataFrame:
    """Interpretable coordinates: share of shots at the rim and from three."""
    z = zones.fillna(0)
    return pd.DataFrame(
        {
            "rim_share": z.get("zone_rim", 0.0),
            "three_share": z.get("zone_corner_three", 0.0) + z.get("zone_above_break_three", 0.0),
        },
        index=zones.index,
    )


def style_spaces(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """College and NBA style weights in one NMF space fit on college maps."""
    college = read_table("modeled", "features", "shot_maps", s).set_index("bbref_id")
    model = shot_xy.fit_styles(college, k=STYLE_K)
    nba = nba_maps(s)
    return shot_xy.style_weights(model, college), shot_xy.style_weights(model, nba), model


def plays_like(
    college_w: pd.DataFrame,
    nba_w: pd.DataFrame,
    heights: pd.Series,
    subjects: pd.Index,
    use_nba_for: pd.Index,
) -> pd.DataFrame:
    """For each subject: nearest NBA early-career style mixes (cosine similarity),
    excluding himself, within MAX_HEIGHT_DIFF inches."""
    pool = nba_w.to_numpy()
    pool_n = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    pool_h = heights.reindex(nba_w.index).to_numpy(dtype=float)
    rows = []
    for pid in subjects:
        src = nba_w if pid in use_nba_for and pid in nba_w.index else college_w
        if pid not in src.index:
            continue
        v = src.loc[pid].to_numpy()
        v = v / np.linalg.norm(v)
        sim = pool_n @ v
        h = float(heights.get(pid, np.nan))
        ok = (nba_w.index != pid) & (
            np.isnan(h) | np.isnan(pool_h) | (np.abs(pool_h - h) <= MAX_HEIGHT_DIFF)
        )
        order = [j for j in np.argsort(-sim) if ok[j]][:N_PLAYS_LIKE]
        for rank, j in enumerate(order, 1):
            rows.append(
                {
                    "bbref_id": pid,
                    "rank": rank,
                    "plays_like_id": nba_w.index[j],
                    "style_similarity": float(sim[j]),
                    "basis": "NBA shots" if src is nba_w else "college shots",
                }
            )
    return pd.DataFrame(rows)


def similarity_percentile(nba_w: pd.DataFrame, sims: pd.Series) -> pd.Series:
    """Share of all NBA player pairs that are less similar than each score. Raw cosine
    similarities run high (two random players are ~0.82 alike), so this is the honest scale."""
    x = nba_w.to_numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    pairs = np.sort((x @ x.T)[np.triu_indices(len(x), 1)])
    return pd.Series(np.searchsorted(pairs, sims.to_numpy()) / len(pairs), index=sims.index)


def style_map(
    college_w: pd.DataFrame,
    nba_w: pd.DataFrame,
    labels: dict[str, str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    both = pd.concat([college_w.assign(source="college"), nba_w.assign(source="NBA early career")])
    x = both.drop(columns="source").to_numpy()
    xy = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca").fit_transform(x)
    out = both[["source"]].copy()
    out["x"], out["y"] = xy[:, 0], xy[:, 1]
    out["dominant_style"] = both.drop(columns="source").idxmax(axis=1)
    if labels:
        out["dominant_style"] = out["dominant_style"].map(labels)
    return out.reset_index(names="bbref_id")


def run(s: Settings) -> None:
    write_table(trajectory_bands(s), "modeled", "grading", "trajectory_bands", s)
    from draft_dna.features.shot_styles import describe_styles

    college_w, nba_w, model = style_spaces(s)
    labels = describe_styles(model).set_index("style")["label"].to_dict()
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    feats = read_table("modeled", "features", "predraft", s).set_index("bbref_id")
    heights = feats["height_in"].astype(float)
    recent = players.index[players["drafted"] & (players["draft_year"] >= 2022)]
    pl = plays_like(college_w, nba_w, heights, recent, use_nba_for=nba_w.index)
    pl["plays_like_name"] = pl["plays_like_id"].map(players["player_name"])
    pl["style_percentile"] = similarity_percentile(nba_w, pl["style_similarity"])
    write_table(pl, "modeled", "grading", "plays_like", s)
    write_table(college_w.reset_index(names="bbref_id"), "modeled", "grading", "styles_college", s)
    write_table(nba_w.reset_index(names="bbref_id"), "modeled", "grading", "styles_nba", s)
    write_table(nba_maps(s).reset_index(names="bbref_id"), "modeled", "grading", "nba_maps", s)
    sm = style_map(college_w, nba_w, labels)
    sm["player_name"] = sm["bbref_id"].map(players["player_name"])
    college_axes = readable_axes(
        read_table("modeled", "features", "shot_zones", s).set_index("bbref_id")
    )
    nba_axes = readable_axes(nba_zone_shares(s))
    is_college = sm["source"] == "college"
    for col in ("rim_share", "three_share"):
        sm[col] = np.where(
            is_college,
            sm["bbref_id"].map(college_axes[col]),
            sm["bbref_id"].map(nba_axes[col]),
        )
    write_table(sm, "modeled", "grading", "style_map", s)
    log.info("extras: %d plays-like rows, %d players on the style map", len(pl), len(sm))
