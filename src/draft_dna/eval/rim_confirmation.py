"""Pre-registered rim-finishing confirmation test (DECISIONS D029), run only when its data
exists. Nothing here is computed early: until a cohort class has year-4 outcomes, only the
status (cohort size, when each class becomes testable) is written.

Exactly as fixed in D029:
- cohort: drafted college players from the 2023-2026 classes (never used in Phase 5)
- outcome: best 3-season value through year 4 (Phase 5 design B)
- model: LightGBM pick + stats, Phase 5's frozen configuration, trained on classes <= 2022
- test 1: Spearman rho between rim_fg_eb_rel and that model's residual, 95% CI > 0
- test 2: CRPS of pick + stats + rim_fg_eb_rel vs pick + stats, improvement CI excludes 0
- confirmed only if both pass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from draft_dna.config import Settings
from draft_dna.eval import metrics as M
from draft_dna.eval import phase5 as P5
from draft_dna.features.predraft import STATS_FEATURES
from draft_dna.ingest.storage import write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

COHORT = (2023, 2026)
TRAIN_MAX = 2022
FEATURE = "rim_fg_eb_rel"
HORIZON = 4
N_BOOT = 2000


def status(df: pd.DataFrame, last_complete: int) -> pd.DataFrame:
    rows = []
    for c in range(COHORT[0], COHORT[1] + 1):
        cls = df[(df["draft_year"] == c) & df[FEATURE].notna()]
        ready = c + HORIZON <= last_complete
        rows.append(
            {
                "draft_class": c,
                "players": len(cls),
                "ready": ready,
                "testable_after": f"{c + HORIZON - 1}-{str(c + HORIZON)[-2:]} season",
            }
        )
    return pd.DataFrame(rows)


def run_test(df: pd.DataFrame, classes: list[int], train_max: int = TRAIN_MAX) -> dict[str, float]:
    """The D029 tests on `classes` (also usable as a dry run on older classes)."""
    base_feats = ["log_pick", *STATS_FEATURES]
    train = df[(df["draft_year"] <= train_max) & df["y_peak4"].notna()]
    test = df[df["draft_year"].isin(classes) & df["y_peak4"].notna() & df[FEATURE].notna()]
    y_tr, y = train["y_peak4"].to_numpy(), test["y_peak4"].to_numpy()
    base = P5._lgbm(base_feats, "pick+stats").fit(train, y_tr)
    plus = P5._lgbm([*base_feats, FEATURE], "pick+stats+rim").fit(train, y_tr)
    qb = M.monotone(base.predict_quantiles(test))
    qp = M.monotone(plus.predict_quantiles(test))
    resid = y - qb[:, M.qidx(M.MEDIAN)]
    x = test[FEATURE].to_numpy()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], resid[i]).correlation)
    crps_b = pd.Series(M.crps(y, qb), index=test["bbref_id"])
    crps_p = pd.Series(M.crps(y, qp), index=test["bbref_id"])
    diff, lo, hi = P5.paired(crps_p, crps_b, n_boot=N_BOOT)
    rho = float(spearmanr(x, resid).correlation)
    rho_lo, rho_hi = (float(v) for v in np.nanpercentile(boots, [2.5, 97.5]))
    return {
        "n": float(len(test)),
        "rho": rho,
        "rho_lo": rho_lo,
        "rho_hi": rho_hi,
        "crps_diff": diff,
        "crps_lo": lo,
        "crps_hi": hi,
        "confirmed": float(rho_lo > 0 and hi < 0),
    }


def run(s: Settings) -> None:
    df = P5.frame(s)
    last_complete = s.current_nba_season - 1
    st = status(df, last_complete)
    write_table(st, "modeled", "rim_test", "status", s)
    ready = st.loc[st["ready"], "draft_class"].tolist()
    if not ready:
        log.info(
            "rim confirmation test: waiting (first class testable %s)", st["testable_after"].iloc[0]
        )
        return
    result = run_test(df, ready)
    write_table(
        pd.DataFrame([{**result, "classes": ",".join(map(str, ready))}]),
        "modeled",
        "rim_test",
        "result",
        s,
    )
    log.info("rim confirmation test on %s: %s", ready, result)
