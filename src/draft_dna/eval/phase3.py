"""Phase 3 evaluation: run every model through the rolling backtest and score it.

Model selection was done on the 2006-2012 test years only ("tuning"); 2013-2020 is the
holdout, touched once for the final comparison (DECISIONS D020). Predictions are cached
under data/modeled/backtest/ so reports don't refit models.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import backtest as bt
from draft_dna.eval import metrics as M
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger
from draft_dna.models import distribution as D
from draft_dna.models import knn

log = get_logger(__name__)

TUNING = (2006, 2012)
HOLDOUT = (2013, 2020)
ALLSTAR_CUT = 1.76  # tier cutoffs from Phase 2 (params.json); asserted in tests
BUST_CUT = 0.32


def lgbm_reg() -> D.LgbmQuantile:
    return D.LgbmQuantile(n_estimators=150, min_child_samples=40, num_leaves=5)


def bayes() -> D.BayesTobit:
    return D.BayesTobit(transform="sqrt")


def final_blend() -> D.QuantileBlend:
    return D.QuantileBlend([knn.PickBaseline, lgbm_reg, bayes], [1, 1, 1], "Final blend")


def pick_conformal() -> D.Conformal:
    return D.Conformal(knn.PickBaseline)


MODELS: dict[str, Callable[[], Any]] = {
    "Pick only": knn.PickBaseline,
    "Pick only + conformal": pick_conformal,
    "Stats kNN": knn.StatsKnn,
    "Learned-weight kNN": knn.SupervisedWeightKnn,
    "Tree-proximity kNN": knn.TreeProximityKnn,
    "LightGBM quantile": lgbm_reg,
    "NGBoost": D.NgbQuantile,
    "Bayesian hierarchical": bayes,
    "Final blend": final_blend,
    "Final blend + conformal": lambda: D.Conformal(final_blend),
}
BASELINES = ("Pick only", "Stats kNN")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def predictions(s: Settings, df: pd.DataFrame, refresh: bool = False) -> dict[str, pd.DataFrame]:
    warnings.filterwarnings("ignore")
    logging.getLogger("pymc").setLevel(logging.ERROR)
    out = {}
    for name, make in MODELS.items():
        path = table_path("modeled", "backtest", _slug(name), s)
        if path.exists() and not refresh:
            out[name] = pd.read_parquet(path)
            continue
        log.info("backtesting %s", name)
        out[name] = bt.run_backtest(df, make)
        write_table(out[name], "modeled", "backtest", _slug(name), s)
    return out


def per_player_scores(pred: pd.DataFrame) -> pd.DataFrame:
    q = bt.qmatrix(pred)
    y = pred["y"].to_numpy()
    p_star = 1 - M.cdf_at(q, ALLSTAR_CUT)
    p_bust = M.cdf_at(q, BUST_CUT)
    return pd.DataFrame(
        {
            "bbref_id": pred["bbref_id"],
            "draft_year": pred["draft_year"],
            "crps": M.crps(y, q),
            "pinball_floor": M.pinball(y, q[:, M.qidx(M.FLOOR)], M.FLOOR),
            "pinball_median": M.pinball(y, q[:, M.qidx(M.MEDIAN)], M.MEDIAN),
            "pinball_ceiling": M.pinball(y, q[:, M.qidx(M.CEILING)], M.CEILING),
            "brier_allstar": M.brier(p_star, y >= ALLSTAR_CUT),
            "brier_bust": M.brier(p_bust, y < BUST_CUT),
            "below_floor": (y < q[:, M.qidx(M.FLOOR)]) + 0.5 * (y == q[:, M.qidx(M.FLOOR)]),
            "above_ceiling": (y > q[:, M.qidx(M.CEILING)]) + 0.5 * (y == q[:, M.qidx(M.CEILING)]),
        }
    )


SCORES = ["crps", "pinball_floor", "pinball_median", "pinball_ceiling", "brier_allstar",
          "brier_bust"]  # fmt: skip


def summarize(preds: dict[str, pd.DataFrame], years: tuple[int, int], n_boot: int = 1000,
              seed: int = 0) -> pd.DataFrame:  # fmt: skip
    """Mean score per model, plus paired-bootstrap 95% CIs of the difference vs each
    baseline (negative = better than the baseline)."""
    scores = {n: per_player_scores(p).set_index("bbref_id") for n, p in preds.items()}
    ids = scores["Pick only"].query(f"{years[0]} <= draft_year <= {years[1]}").index
    rng = np.random.default_rng(seed)
    boot = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]
    rows = []
    for name, sc in scores.items():
        sc = sc.loc[ids]
        row: dict[str, Any] = {"model": name, "n": len(ids)}
        for m in SCORES:
            row[m] = sc[m].mean()
        row["below_floor"] = sc["below_floor"].mean()
        row["above_ceiling"] = sc["above_ceiling"].mean()
        for base in BASELINES:
            diff = (sc["crps"] - scores[base].loc[ids, "crps"]).to_numpy()
            bs = np.array([diff[b].mean() for b in boot])
            key = "pick" if base == "Pick only" else "knn"
            row[f"crps_vs_{key}"] = diff.mean()
            row[f"crps_vs_{key}_lo"] = np.percentile(bs, 2.5)
            row[f"crps_vs_{key}_hi"] = np.percentile(bs, 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_curve(pred: pd.DataFrame) -> pd.DataFrame:
    """Observed share below each predicted quantile level (calibration)."""
    q = bt.qmatrix(pred)
    y = pred["y"].to_numpy()
    return pd.DataFrame(
        {"level": M.QS, "observed": [M.coverage_below(y, q[:, i]) for i in range(len(M.QS))]}
    )


def reliability(pred: pd.DataFrame, which: str, bins: int = 8) -> pd.DataFrame:
    q = bt.qmatrix(pred)
    y = pred["y"].to_numpy()
    if which == "allstar":
        p, hit = 1 - M.cdf_at(q, ALLSTAR_CUT), y >= ALLSTAR_CUT
    else:
        p, hit = M.cdf_at(q, BUST_CUT), y < BUST_CUT
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return (
        pd.DataFrame({"bin": b, "p": p, "hit": hit})
        .groupby("bin")
        .agg(predicted=("p", "mean"), observed=("hit", "mean"), n=("hit", "size"))
        .reset_index(drop=True)
    )


def load_frame(s: Settings) -> pd.DataFrame:
    return bt.modeling_frame(s)


def tier_cuts(s: Settings) -> list[float]:
    import json

    params = json.loads(
        table_path("modeled", "outcomes", "params", s).with_suffix(".json").read_text()
    )
    return list(params["tier_cuts_peak3"])


def players(s: Settings) -> pd.DataFrame:
    return read_table("modeled", "core", "players", s)
