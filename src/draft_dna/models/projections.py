"""Per-player Phase 3 output: outcome band, tier probabilities and top comps.

Every projection is *as of draft night*: the model for class Y is trained only on classes
whose 6-season outcome was complete by then (c + 6 <= Y), exactly as in the backtest.
Classes before 2002 have no such history and get no projection.

- Outcome band and tier probabilities: the model of record, draft-slot history with
  conformal calibration (DECISIONS D021).
- Comps: standardized pre-draft stats kNN with the feature-overlap guard (D022). The comp
  pool is every earlier class whose 6-season outcome is known today; comps describe a
  statistical profile and their outcomes are context, not the forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import backtest as bt
from draft_dna.eval import metrics as M
from draft_dna.eval.phase3 import pick_conformal
from draft_dna.features.predraft import STATS_FEATURES
from draft_dna.ingest.storage import write_table
from draft_dna.logging_utils import get_logger
from draft_dna.models.knn import StatsKnn
from draft_dna.outcomes.tiers import TIERS, tier_cuts, tier_probabilities

log = get_logger(__name__)

N_COMPS = 15
FIRST_PROJECTED_CLASS = 2002
FEATURE_LABELS = {
    "age_at_draft": "age", "height_in": "height", "weight_lb": "weight",
    "wingspan_in": "wingspan", "wingspan_minus_height": "length (wingspan - height)",
    "standing_reach_in": "standing reach", "max_vertical": "vertical", "lane_agility": "agility",
    "sprint": "sprint", "recruit_rank_top100": "recruiting rank",
    "college_seasons": "years in college",
    "games": "games", "mpg": "minutes", "pts_per40": "scoring", "trb_per40": "rebounding",
    "ast_per40": "passing", "stl_per40": "steals", "blk_per40": "blocks", "tov_per40": "turnovers",
    "ts_pct": "efficiency (TS%)", "ts_rel": "efficiency vs era", "fg3a_rate": "3PT volume",
    "fg3a_rate_rel": "3PT volume vs era", "fg3_pct": "3PT%", "fg3a_per40": "3PA per 40",
    "ft_pct": "FT%", "career_ft_pct": "career FT%", "ftr": "free-throw rate", "usg_pct": "usage",
    "ast_pct": "assist rate", "tov_pct": "turnover rate", "trb_pct": "rebound rate",
    "blk_pct": "block rate", "stl_pct": "steal rate", "bpm": "BPM", "obpm": "OBPM", "dbpm": "DBPM",
    "team_srs": "team strength", "team_sos": "schedule strength",
}  # fmt: skip
# Features shown as "why these two are similar" (flags and trivially shared ones excluded:
# every one-and-done freshman has the same years in college).
TRIVIAL = ("college_seasons", "games")
DESCRIPTIVE = [f for f in STATS_FEATURES if not f.startswith(("src_", "pos_")) and f not in TRIVIAL]


def _driving_features(model: StatsKnn, row: pd.DataFrame, comp_idx: int, n: int = 3) -> str:
    gaps = model.feature_gaps(row, comp_idx).reindex(DESCRIPTIVE).dropna()
    return ", ".join(FEATURE_LABELS.get(f, f) for f in gaps.nsmallest(n).index)


def build(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = bt.modeling_frame(s)
    cuts = tier_cuts(s)
    known = df[df[bt.TARGET].notna()]
    proj_rows, comp_rows = [], []
    for year in sorted(df["draft_year"].dropna().unique().astype(int)):
        cls = df[df["draft_year"] == year]
        train = known[known["draft_year"] + bt.HORIZON <= year]
        if year >= FIRST_PROJECTED_CLASS and len(train) > 0:
            model = pick_conformal().fit(train, train[bt.TARGET].to_numpy())
            q = M.monotone(model.predict_quantiles(cls))
            tp = tier_probabilities(q, cuts)
            out = pd.DataFrame(
                {
                    "bbref_id": cls["bbref_id"].to_numpy(),
                    "floor": q[:, M.qidx(M.FLOOR)],
                    "median": q[:, M.qidx(M.MEDIAN)],
                    "ceiling": q[:, M.qidx(M.CEILING)],
                    "p_allstar_plus": tp[:, 4] + tp[:, 5],
                    "p_bust_or_worse": tp[:, 0] + tp[:, 1],
                    "n_train": len(train),
                    "max_train_class": int(train["draft_year"].max()),
                }
            )
            for i, t in enumerate(TIERS):
                out[f"p_{t.lower().replace(' ', '_').replace('-', '_')}"] = tp[:, i]
            for i, tau in enumerate(M.QS):
                out[f"q{tau:.2f}"] = q[:, i]
            proj_rows.append(out)

        pool = known[known["draft_year"] < year]
        if len(pool) < N_COMPS:
            continue
        comps = StatsKnn(k=N_COMPS).fit(pool, pool[bt.TARGET].to_numpy())
        idx, dist = comps.neighbors(cls)
        all_d = comps._distances(cls)
        for r in range(len(cls)):
            row = cls.iloc[[r]]
            finite = all_d[r][np.isfinite(all_d[r])]
            for rank, (j, d) in enumerate(zip(idx[r], dist[r], strict=True), start=1):
                comp_rows.append(
                    {
                        "bbref_id": row["bbref_id"].iloc[0],
                        "rank": rank,
                        "comp_id": pool.iloc[j]["bbref_id"],
                        "comp_name": pool.iloc[j]["player_name"],
                        "comp_draft_year": int(pool.iloc[j]["draft_year"]),
                        "comp_pick": int(pool.iloc[j]["pick"]),
                        # Similarity 0-100: 100 = identical profile, 0 = as far as the
                        # prospect's typical (median) distance to the pool.
                        "similarity": round(max(0.0, 100 * (1 - d / np.median(finite))), 1),
                        "low_overlap": bool(d >= StatsKnn.OVERLAP_PENALTY),
                        "comp_peak6": float(pool.iloc[j][bt.TARGET]),
                        "driving_features": _driving_features(comps, row, j),
                    }
                )
    proj = pd.concat(proj_rows, ignore_index=True)
    names = df.set_index("bbref_id")[["player_name", "draft_year", "pick", "prospect_source",
                                      "position", bt.TARGET]]  # fmt: skip
    proj = names.join(proj.set_index("bbref_id"), how="inner").reset_index()
    return proj, pd.DataFrame(comp_rows)


def run(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    proj, comps = build(s)
    write_table(proj, "modeled", "projections", "projections", s)
    write_table(comps, "modeled", "projections", "comps", s)
    log.info("projections: %d players; comps: %d rows", len(proj), len(comps))
    return proj, comps
