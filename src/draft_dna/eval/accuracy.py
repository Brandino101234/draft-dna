"""Tables for the app's "How accurate is this?" page, from saved backtest predictions.

Everything here is held out: Phase 3 predictions for the 2013-2020 classes were made by
models trained only on earlier classes and were never used to choose a model (D021).
The target is the Phase 3 one: best 3-season value through year 6.

- models: CRPS and coverage for the model of record and the alternatives it beat or tied
- calibration: for each predicted percentile, the share of careers that finished below it
- allstar_reliability: predicted chance of All-Star-or-better vs how often it happened
- sharpening: grading validation (how the range tightens as seasons are played)
"""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import metrics as M
from draft_dna.eval import phase3 as P
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger
from draft_dna.outcomes.tiers import TIERS, tier_cuts, tier_probabilities

log = get_logger(__name__)

RECORD = "Pick only + conformal"
MODELS = {  # Phase 3 model name -> label on the page
    RECORD: "Draft slot + calibration (model of record)",
    "Pick only": "Draft slot, uncalibrated",
    "Final blend + conformal": "Best stats blend (slot + LightGBM + Bayesian)",
    "Bayesian hierarchical": "Bayesian hierarchical (stats)",
    "LightGBM quantile": "LightGBM (stats)",
    "Stats kNN": "Stats comps (nearest neighbors)",
}
ALLSTAR_BINS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]


def load_predictions(s: Settings) -> dict[str, pd.DataFrame]:
    out = {}
    for name in MODELS:
        path = table_path("modeled", "backtest", P._slug(name), s)
        if path.exists():
            out[name] = pd.read_parquet(path)
    return out


def _holdout(pred: pd.DataFrame) -> pd.DataFrame:
    lo, hi = P.HOLDOUT
    return pred[pred["draft_year"].between(lo, hi) & pred["y"].notna()]


def model_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t = P.summarize(preds, P.HOLDOUT)
    t["label"] = t["model"].map(MODELS)
    keep = ["model", "label", "crps", "crps_vs_pick", "crps_vs_pick_lo", "crps_vs_pick_hi"]
    keep += ["below_floor", "above_ceiling", "brier_allstar"]
    return t[[c for c in keep if c in t.columns]].sort_values("crps").reset_index(drop=True)


def calibration(pred: pd.DataFrame) -> pd.DataFrame:
    """Share of held-out careers below each predicted percentile (midpoint for ties at 0)."""
    h = _holdout(pred)
    y = h["y"].to_numpy()
    rows = []
    for level, col in zip(M.QS, M.QCOLS, strict=True):
        q = h[col].to_numpy()
        below = ((y < q).mean() + (y <= q).mean()) / 2
        rows.append({"level": float(level), "observed": float(below), "n": len(h)})
    return pd.DataFrame(rows)


def allstar_reliability(pred: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    h = _holdout(pred)
    p = tier_probabilities(h[M.QCOLS].to_numpy(), cuts)[:, TIERS.index("All-Star") :].sum(axis=1)
    hit = h["y"].to_numpy() >= cuts[3]
    bins = pd.cut(p, ALLSTAR_BINS, include_lowest=True)
    t = pd.DataFrame({"bin": bins, "p": p, "hit": hit}).groupby("bin", observed=True)
    out = t.agg(predicted=("p", "mean"), observed=("hit", "mean"), n=("hit", "size"))
    out = out.reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def run(s: Settings) -> None:
    preds = load_predictions(s)
    if RECORD not in preds:
        log.warning("accuracy: no saved backtest predictions; run `draft-dna backtest` first")
        return
    write_table(model_table(preds), "modeled", "accuracy", "models", s)
    write_table(calibration(preds[RECORD]), "modeled", "accuracy", "calibration", s)
    write_table(
        allstar_reliability(preds[RECORD], tier_cuts(s)),
        "modeled",
        "accuracy",
        "allstar_reliability",
        s,
    )
    v = read_table("modeled", "grading", "validation", s)
    write_table(v[v["quantiles"] == "calibrated"], "modeled", "accuracy", "sharpening", s)
    log.info("accuracy tables written (%d held-out players)", len(_holdout(preds[RECORD])))
