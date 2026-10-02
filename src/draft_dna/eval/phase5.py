"""Phase 5: does shot data help? Implements the pre-registered plan in DECISIONS D027.

Design B (primary): year-4 peak, strict leakage rule (class c trains Y only if c + 4 <= Y).
Design A (sensitivity): year-6 peak, trains on any class drafted before Y.
Every model sees the same cohort: drafted college players with >= 100 shot-type FGA.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import backtest as bt
from draft_dna.eval import metrics as M
from draft_dna.eval.phase3 import ALLSTAR_CUT, BUST_CUT, pick_conformal
from draft_dna.features.predraft import STATS_FEATURES
from draft_dna.features.shot_dna import ASSIST_FEATURES, SHOT_FEATURES
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger
from draft_dna.models import distribution as D
from draft_dna.models.knn import GuardedKnn, StatsKnn, _NeighborModel

log = get_logger(__name__)

MIN_SHOT_FGA = 100
SHOT = SHOT_FEATURES + ASSIST_FEATURES
FEATURE_SETS = {"stats": STATS_FEATURES, "shot": SHOT, "stats+shot": STATS_FEATURES + SHOT}


@dataclass(frozen=True)
class Design:
    name: str
    target: str
    lag: int
    first: int
    last: int
    tune: tuple[int, int]  # years used to choose the comp blend weight
    report: tuple[int, int]  # years where tuned models are reported


DESIGNS = {
    "B": Design("B: strict, year-4 outcome", "y_peak4", 4, 2014, 2022, (2014, 2017), (2018, 2022)),
    "A": Design(
        "A: drafted-before-Y, year-6 outcome", "y_peak6", 1, 2012, 2020, (2012, 2015), (2016, 2020)
    ),
}


def frame(s: Settings) -> pd.DataFrame:
    df = bt.modeling_frame(s)
    shots = read_table("modeled", "features", "shot_dna", s).set_index("bbref_id")
    cols = ["shot_fga", *SHOT]
    df = df.drop(columns=[c for c in cols if c in df]).join(shots[cols], on="bbref_id")
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    df["y_peak4"] = df["bbref_id"].map(otn[otn["n"] == 4].set_index("bbref_id")["peak3_blend"])
    cohort = (df["src_college"] == 1) & (df["shot_fga"] >= MIN_SHOT_FGA)
    return df[cohort].reset_index(drop=True)


class CombinedKnn(_NeighborModel):
    """Comps on a weighted mix of stats distance and shot distance (each divided by its
    median, so the weight w is on comparable scales). w = 1 is stats only, w = 0 shot only.
    The shot part carries the size/position guardrail."""

    name = "Combined kNN"

    def __init__(self, w: float, k: int | None = None) -> None:
        super().__init__(k)
        self.w = w

    def _fit_space(self, train: pd.DataFrame, y: np.ndarray) -> None:
        self.stats = StatsKnn(features=STATS_FEATURES).fit(train, y)
        self.shot = GuardedKnn(features=SHOT).fit(train, y)

    def _distances(self, test: pd.DataFrame) -> np.ndarray:
        ds = self.stats._distances(test)
        dh = self.shot._distances(test)
        ds = ds / np.median(ds[ds < StatsKnn.OVERLAP_PENALTY])
        dh = dh / np.median(dh[dh < StatsKnn.OVERLAP_PENALTY])
        return self.w * ds + (1 - self.w) * dh


def _lgbm(features: list[str], label: str) -> D.LgbmQuantile:
    # Phase 3's regularized configuration, frozen (no tuning in Phase 5).
    return D.LgbmQuantile(
        features=features, label=label, n_estimators=150, min_child_samples=40, num_leaves=5
    )


def _stats_knn() -> StatsKnn:
    return StatsKnn(features=STATS_FEATURES)


def _shot_knn() -> GuardedKnn:
    return GuardedKnn(features=SHOT)


def models() -> dict[str, Callable[[], Any]]:
    out: dict[str, Callable[[], Any]] = {
        "Pick only + conformal": pick_conformal,
        "kNN stats": _stats_knn,
        "kNN shot": _shot_knn,
    }
    for w in (0.25, 0.5, 0.75):
        out[f"kNN stats+shot w={w}"] = partial(CombinedKnn, w)
    for name, feats in FEATURE_SETS.items():
        out[f"LightGBM {name}"] = partial(_lgbm, feats, f"LightGBM {name}")
        out[f"LightGBM pick+{name}"] = partial(_lgbm, ["log_pick", *feats], f"LightGBM pick+{name}")
    return out


def predictions(s: Settings, design_key: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    warnings.filterwarnings("ignore")
    logging.getLogger("pymc").setLevel(logging.ERROR)
    d = DESIGNS[design_key]
    df = frame(s)
    out = {}
    for name, make in models().items():
        slug = f"p5{design_key}_" + name.lower().replace(" ", "_").replace("+", "plus").replace(
            "=", ""
        )
        path = table_path("modeled", "backtest", slug, s)
        if path.exists() and not refresh:
            out[name] = pd.read_parquet(path)
            continue
        log.info("phase 5 design %s: %s", design_key, name)
        out[name] = bt.run_backtest(df, make, d.first, d.last, target=d.target, lag=d.lag)
        write_table(out[name], "modeled", "backtest", slug, s)
    return out


def scores(pred: pd.DataFrame) -> pd.DataFrame:
    q = bt.qmatrix(pred)
    y = pred["y"].to_numpy()
    return pd.DataFrame(
        {
            "bbref_id": pred["bbref_id"],
            "draft_year": pred["draft_year"],
            "crps": M.crps(y, q),
            "pinball_floor": M.pinball(y, q[:, M.qidx(M.FLOOR)], M.FLOOR),
            "pinball_median": M.pinball(y, q[:, M.qidx(M.MEDIAN)], M.MEDIAN),
            "pinball_ceiling": M.pinball(y, q[:, M.qidx(M.CEILING)], M.CEILING),
            "brier_bust": M.brier(M.cdf_at(q, BUST_CUT), y < BUST_CUT),
            "brier_allstar": M.brier(1 - M.cdf_at(q, ALLSTAR_CUT), y >= ALLSTAR_CUT),
            "below_floor": (y < q[:, M.qidx(M.FLOOR)]) + 0.5 * (y == q[:, M.qidx(M.FLOOR)]),
            "above_ceiling": (y > q[:, M.qidx(M.CEILING)]) + 0.5 * (y == q[:, M.qidx(M.CEILING)]),
        }
    ).set_index("bbref_id")


def paired(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) over players with a 95% bootstrap CI (negative = a is better)."""
    diff = (a - b.reindex(a.index)).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summary(preds: dict[str, pd.DataFrame], years: tuple[int, int]) -> pd.DataFrame:
    rows = []
    sc = {n: scores(p) for n, p in preds.items()}
    base = sc["Pick only + conformal"]
    for name, s in sc.items():
        s = s[s["draft_year"].between(*years)]
        diff = paired(s["crps"], base["crps"])
        rows.append(
            {
                "model": name,
                "n": len(s),
                **s.drop(columns="draft_year").mean().to_dict(),
                "crps_vs_pick": diff[0],
                "ci_lo": diff[1],
                "ci_hi": diff[2],
            }
        )
    return pd.DataFrame(rows)


def hypotheses(preds: dict[str, pd.DataFrame], years: tuple[int, int]) -> pd.DataFrame:
    sc = {n: scores(p) for n, p in preds.items()}
    sc = {n: s[s["draft_year"].between(*years)]["crps"] for n, s in sc.items()}
    tests = [
        ("H1: pick+stats+shot vs pick+stats", "LightGBM pick+stats+shot", "LightGBM pick+stats"),
        ("H2a: shot vs stats (no pick)", "LightGBM shot", "LightGBM stats"),
        ("H2b: stats+shot vs stats (no pick)", "LightGBM stats+shot", "LightGBM stats"),
        ("H2c: kNN shot vs kNN stats", "kNN shot", "kNN stats"),
        ("H3: pick+stats+shot vs pick only", "LightGBM pick+stats+shot", "Pick only + conformal"),
        ("H3: pick+shot vs pick only", "LightGBM pick+shot", "Pick only + conformal"),
    ]
    rows = []
    for label, a, b in tests:
        m, lo, hi = paired(sc[a], sc[b])
        rows.append(
            {
                "comparison": label,
                "crps_diff": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "verdict": "better"
                if hi < 0
                else ("worse" if lo > 0 else "no detectable difference"),
            }
        )
    return pd.DataFrame(rows)


def tune_comp_weight(preds: dict[str, pd.DataFrame], d: Design) -> tuple[float, pd.DataFrame]:
    """Choose the stats/shot blend weight on the tuning years only."""
    rows = []
    for w, name in (
        (1.0, "kNN stats"),
        (0.75, "kNN stats+shot w=0.75"),
        (0.5, "kNN stats+shot w=0.5"),
        (0.25, "kNN stats+shot w=0.25"),
        (0.0, "kNN shot"),
    ):
        s = scores(preds[name])
        rows.append(
            {
                "w_stats": w,
                "model": name,
                "crps_tune": s[s["draft_year"].between(*d.tune)]["crps"].mean(),
                "crps_report": s[s["draft_year"].between(*d.report)]["crps"].mean(),
            }
        )
    t = pd.DataFrame(rows)
    return float(t.loc[t["crps_tune"].idxmin(), "w_stats"]), t


def subgroups(
    preds: dict[str, pd.DataFrame], s: Settings, years: tuple[int, int], a: str, b: str
) -> pd.DataFrame:
    df = frame(s).set_index("bbref_id")
    sa, sb = scores(preds[a]), scores(preds[b])
    sa = sa[sa["draft_year"].between(*years)]
    band = pd.cut(df["pick"], [0, 14, 30, 60], labels=["picks 1-14", "picks 15-30", "picks 31-60"])
    rows = []
    for label, groups in (("position", df["position"]), ("pick band", band.astype(str))):
        for g in sorted(groups.dropna().unique()):
            ids = sa.index[sa.index.map(groups) == g]
            if len(ids) < 20:
                continue
            m, lo, hi = paired(sa.loc[ids, "crps"], sb["crps"])
            rows.append(
                {
                    "split": label,
                    "group": g,
                    "n": len(ids),
                    "crps_diff": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- tier B (spatial)
ZONE_COLS = [
    "zone_rim",
    "zone_short_mid",
    "zone_long_mid",
    "zone_corner_three",
    "zone_above_break_three",
]
STYLE_K = 6


class SpatialLgbm:
    """LightGBM pick+stats plus shot-location features: 5 zone shares and NMF style
    weights. NMF is refit on each fold's training players only (no leakage)."""

    name = "LightGBM pick+stats+spatial"

    def __init__(self, maps: pd.DataFrame) -> None:
        self.maps = maps

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        from draft_dna.features import shot_xy

        m = self.maps.reindex(df["bbref_id"])
        w = shot_xy.style_weights(self.nmf, m.fillna(0.0))
        w[m.filter(regex=r"^c\d+$").isna().all(axis=1).to_numpy()] = np.nan
        out = df.reset_index(drop=True).copy()
        for c in w.columns:
            out[c] = w[c].to_numpy()
        return out

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> SpatialLgbm:
        from draft_dna.features import shot_xy

        tm = self.maps.loc[self.maps.index.intersection(train["bbref_id"])]
        self.nmf = shot_xy.fit_styles(tm, k=STYLE_K)
        feats = ["log_pick", *STATS_FEATURES, *ZONE_COLS, *[f"style_{i}" for i in range(STYLE_K)]]
        self.model = _lgbm(feats, self.name).fit(self._features(train), y)
        return self

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_quantiles(self._features(test))


def spatial(s: Settings, design_key: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Exploratory: coordinate-eligible cohort, pick+stats vs pick+stats+spatial."""
    d = DESIGNS[design_key]
    df = frame(s)
    zones = read_table("modeled", "features", "shot_zones", s).set_index("bbref_id")
    maps = read_table("modeled", "features", "shot_maps", s).set_index("bbref_id")
    df = df[df["bbref_id"].isin(maps.index)].join(zones[ZONE_COLS], on="bbref_id")
    first = int(df["draft_year"].min()) + d.lag
    preds = {
        "LightGBM pick+stats": bt.run_backtest(
            df,
            partial(_lgbm, ["log_pick", *STATS_FEATURES], "LightGBM pick+stats"),
            first,
            d.last,
            target=d.target,
            lag=d.lag,
        ),
        "LightGBM pick+stats+spatial": bt.run_backtest(
            df, partial(SpatialLgbm, maps), first, d.last, target=d.target, lag=d.lag
        ),
    }
    a, b = scores(preds["LightGBM pick+stats+spatial"]), scores(preds["LightGBM pick+stats"])
    m, lo, hi = paired(a["crps"], b["crps"])
    res = pd.DataFrame(
        [
            {
                "design": d.name,
                "n_test": len(a),
                "test_years": f"{first}-{d.last}",
                "crps_spatial": a["crps"].mean(),
                "crps_base": b["crps"].mean(),
                "diff": m,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        ]
    )
    return res, preds
