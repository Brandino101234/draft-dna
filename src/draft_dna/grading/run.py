"""Grade every drafted player (2002+) from the draft-night projection and seasons played.

Outputs modeled.grading__grades: status, seasons played, prior and posterior summaries of
the year-8 peak, grade letter, confidence, Year-4 Verdict and Career Grade (Phase 6
definitions). Also validates the updating on held-out classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import backtest as bt
from draft_dna.eval import metrics as M
from draft_dna.eval.phase3 import pick_conformal
from draft_dna.grading import bayes as B
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

FIRST_GRADED_CLASS = 2005  # needs >= 100 classes' worth of year-8 history before it
QCOLS = M.QCOLS


def peaks_wide(s: Settings) -> pd.DataFrame:
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    return otn.pivot(index="bbref_id", columns="n", values="peak3_blend")


def asof_prior_grids(s: Settings, horizon: int = B.FINAL_N) -> pd.DataFrame:
    """Draft-night projected quantile grid of the year-`horizon` peak for every drafted
    player (model of record; class c trains Y only if c + horizon <= Y)."""
    df = bt.modeling_frame(s)
    wide = peaks_wide(s)
    df["y_final"] = df["bbref_id"].map(wide[horizon]) if horizon in wide else np.nan
    known = df[df["y_final"].notna()]
    rows = []
    for year in sorted(df["draft_year"].dropna().unique().astype(int)):
        train = known[known["draft_year"] + horizon <= year]
        if len(train) < 100:
            continue
        cls = df[df["draft_year"] == year]
        q = M.monotone(
            pick_conformal().fit(train, train["y_final"].to_numpy()).predict_quantiles(cls)
        )
        rows.append(pd.DataFrame(q, columns=QCOLS).assign(bbref_id=cls["bbref_id"].to_numpy()))
    return pd.concat(rows, ignore_index=True).set_index("bbref_id")


def calibration(
    wide: pd.DataFrame, priors: pd.DataFrame, meas: dict[int, B.Measurement], ids: pd.Index
) -> dict[int, B.Calibration]:
    """Error calibration for each number of seasons (1..7), fit on players in `ids`."""
    ids = ids.intersection(priors.index)
    m0, s0 = B.prior_from_grid(priors.loc[ids, QCOLS].to_numpy())
    y = wide.loc[ids, B.FINAL_N].to_numpy()
    return {
        n: B.fit_calibration(B.update(m0, s0, wide.loc[ids, n].to_numpy(), meas[n]), y)
        for n in range(1, B.FINAL_N)
    }


def grade_all(s: Settings, meas: dict[int, B.Measurement] | None = None) -> pd.DataFrame:
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    careers = read_table("modeled", "outcomes", "careers", s).set_index("bbref_id")
    wide = peaks_wide(s)
    last_complete = s.current_nba_season - 1
    if meas is None:
        drafted = wide[wide.index.map(players["drafted"]).fillna(False).astype(bool)]
        meas = B.fit_measurement(drafted)
    all_priors = asof_prior_grids(s)
    history = wide.dropna(subset=[B.FINAL_N]).index
    zcal = calibration(wide, all_priors, meas, history)
    priors = all_priors[all_priors.index.map(players["draft_year"]) >= FIRST_GRADED_CLASS]
    grid = priors[QCOLS].to_numpy()
    m0, s0 = B.prior_from_grid(grid)

    out = pd.DataFrame(index=priors.index)
    out["player_name"] = out.index.map(players["player_name"])
    out["draft_year"] = out.index.map(players["draft_year"]).astype(int)
    out["pick"] = out.index.map(players["pick_overall"])
    out["seasons"] = (last_complete - out["draft_year"]).clip(lower=0, upper=B.FINAL_N)
    out["retired"] = out.index.map(careers["ended"]).fillna(False).astype(bool)
    out["status"] = [
        B.status(int(n), bool(r)) for n, r in zip(out["seasons"], out["retired"], strict=True)
    ]
    out["peak_so_far"] = [
        wide.loc[i, n] if n > 0 and i in wide.index else np.nan
        for i, n in zip(out.index, out["seasons"], strict=True)
    ]

    mean, sd, weight = m0.copy(), s0.copy(), np.zeros(len(out))
    for n in range(1, B.FINAL_N):
        rows = (out["seasons"].to_numpy() == n) & ~out["retired"].to_numpy()
        if rows.any():
            post = B.update(m0[rows], s0[rows], out["peak_so_far"].to_numpy()[rows], meas[n])
            mean[rows], sd[rows], weight[rows] = post.mean, post.sd, post.data_weight
    final = (out["seasons"].to_numpy() >= B.FINAL_N) | (
        out["retired"].to_numpy() & (out["seasons"].to_numpy() > 0)
    )
    obs_final = np.where(final, out["peak_so_far"].to_numpy(), np.nan)
    mean[final], sd[final], weight[final] = B.s(obs_final[final]), 0.0, 1.0
    pq = grid.copy()  # 0 seasons: the (already conformal-calibrated) draft-night range
    seasons = out["seasons"].to_numpy()
    for n in range(1, B.FINAL_N):
        rows = (seasons == n) & ~final
        if rows.any():
            pq[rows] = zcal[n].quantiles(B.Posterior(mean[rows], sd[rows], weight[rows]))
    pq[final] = (B.s(obs_final[final]) ** 2)[:, None]
    # Peak value through year 8 can never be below the peak already reached.
    reached = np.nan_to_num(out["peak_so_far"].to_numpy(dtype=float), nan=0.0)
    pq = np.maximum(pq, reached[:, None])

    out["projected_floor"] = grid[:, M.qidx(M.FLOOR)]
    out["projected_median"] = grid[:, M.qidx(M.MEDIAN)]
    out["projected_ceiling"] = grid[:, M.qidx(M.CEILING)]
    out["current_floor"] = pq[:, M.qidx(M.FLOOR)]
    out["current_median"] = pq[:, M.qidx(M.MEDIAN)]
    out["current_ceiling"] = pq[:, M.qidx(M.CEILING)]
    out["data_weight"] = weight
    out["confidence"] = 1 - np.clip(sd / s0, 0, 1)
    out["grade"] = B.grade_letter(out["current_median"].to_numpy(), grid)
    out.loc[out["seasons"] == 0, "grade"] = "-"  # no NBA games yet: projection only

    p6 = read_table("modeled", "phase6", "player_outcomes_vs_projection", s).set_index("bbref_id")
    out["year4_verdict"] = out.index.map(p6["verdict4"]).where(out["seasons"] >= 4)
    out["career_grade_verdict"] = out.index.map(p6["verdict8"]).where(out["seasons"] >= B.FINAL_N)
    return out.join(pd.DataFrame(pq, index=out.index, columns=[f"post_{c}" for c in QCOLS]))


def validate(s: Settings, fit_max_class: int = 2010) -> pd.DataFrame:
    """Held-out check: fit the measurement model on classes <= fit_max_class, then for later
    classes with a known year-8 peak, grade them as if only N seasons had been played and
    score the year-8 forecast (CRPS, coverage) at each N against the prior alone."""
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    wide = peaks_wide(s)
    drafted = wide[wide.index.map(players["drafted"]).fillna(False).astype(bool)]
    dy = drafted.index.map(players["draft_year"])
    meas = B.fit_measurement(drafted[dy <= fit_max_class])
    priors = asof_prior_grids(s)
    test = priors[
        (priors.index.map(players["draft_year"]) > fit_max_class)
        & priors.index.isin(drafted.dropna(subset=[B.FINAL_N]).index)
    ]
    grid = test[QCOLS].to_numpy()
    y = drafted.loc[test.index, B.FINAL_N].to_numpy()
    m0, s0 = B.prior_from_grid(grid)
    fit_ids = drafted[(dy <= fit_max_class)].dropna(subset=[B.FINAL_N]).index
    zcal = calibration(drafted, priors, meas, fit_ids)
    rows = []
    for n in range(0, B.FINAL_N):
        obs = drafted.loc[test.index, n].to_numpy() if n else y
        post = B.update(m0, s0, obs, meas.get(n))
        variants = [("normal", post.quantiles())]
        variants.append(("calibrated", grid if n == 0 else zcal[n].quantiles(post)))
        for label, q in variants:
            rows.append(
                {
                    "seasons": n,
                    "quantiles": label,
                    "n_players": len(y),
                    "data_weight": float(post.data_weight.mean()),
                    "crps": float(M.crps(y, q).mean()),
                    # With a pile of outcomes at exactly 0, calibration means
                    # P(y < floor) <= 25% <= P(y <= floor); report both sides.
                    "below_floor": float((y < q[:, M.qidx(M.FLOOR)]).mean()),
                    "at_or_below_floor": float((y <= q[:, M.qidx(M.FLOOR)]).mean()),
                    "above_ceiling": float((y > q[:, M.qidx(M.CEILING)]).mean()),
                }
            )
    return pd.DataFrame(rows)


def run(s: Settings) -> pd.DataFrame:
    grades = grade_all(s)
    write_table(grades.reset_index(), "modeled", "grading", "grades", s)
    write_table(validate(s), "modeled", "grading", "validation", s)
    log.info("graded %d players: %s", len(grades), grades["status"].value_counts().to_dict())
    return grades
