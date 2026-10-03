"""Grade every drafted player from the draft-night projection and seasons played.

The graded outcome is the best 3-season stretch of value *including playoff value*, with
accolade floors: an All-Star selection guarantees at least an All-Star-tier peak, All-NBA
at least an All-NBA-tier peak (D033). Projections are refit on that same measure.

Classes before 2005 have too little earlier history for an as-of-draft-night projection;
they get a *retrospective* projection (draft-slot model fit on all other classes), shown
on cards but never used for calibration or validation.

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
from draft_dna.eval.phase6 import FRANCHISE
from draft_dna.grading import bayes as B
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger
from draft_dna.models import distribution as D
from draft_dna.models import knn
from draft_dna.outcomes.tiers import TIERS, graded_peak, tier_cuts, tier_probabilities

log = get_logger(__name__)

MIN_TRAINING = 100  # an as-of projection needs this many earlier players with outcomes
QCOLS = M.QCOLS


def peaks_wide(s: Settings) -> pd.DataFrame:
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    otn["peak"] = graded_peak(otn, tier_cuts(s))
    return otn.pivot(index="bbref_id", columns="n", values="peak")


def prior_model(n_train: int) -> D.Conformal:
    """Model of record (draft-slot neighbors + conformal), with the neighbor count capped
    at ~20% of the training players. The tuned k = 60 assumed 300+ players; the earliest
    graded classes (2005-2007) have 115-230, and k = 60 there averaged a #1 pick with
    picks down to ~#30, giving every top-5 pick in 2005 the same projection (D038)."""
    k = min(knn.PickBaseline.k, max(10, round(0.2 * n_train)))
    return D.Conformal(lambda: knn.PickBaseline(k=k))


def asof_prior_grids(
    s: Settings, horizon: int = B.FINAL_N, retrospective: bool = False
) -> pd.DataFrame:
    """Draft-night projected quantile grid of the year-`horizon` peak for every drafted
    player (model of record; class c trains Y only if c + horizon <= Y). With
    `retrospective`, classes lacking that much history are projected from all other
    classes instead (flagged in the `retrospective` column)."""
    df = bt.modeling_frame(s)
    wide = peaks_wide(s)
    df["y_final"] = df["bbref_id"].map(wide[horizon]) if horizon in wide else np.nan
    known = df[df["y_final"].notna()]
    rows = []
    for year in sorted(df["draft_year"].dropna().unique().astype(int)):
        train = known[known["draft_year"] + horizon <= year]
        retro = len(train) < MIN_TRAINING
        if retro:
            if not retrospective:
                continue
            train = known[known["draft_year"] != year]
        cls = df[df["draft_year"] == year]
        q = M.monotone(
            prior_model(len(train)).fit(train, train["y_final"].to_numpy()).predict_quantiles(cls)
        )
        rows.append(
            pd.DataFrame(q, columns=QCOLS).assign(
                bbref_id=cls["bbref_id"].to_numpy(), retrospective=retro
            )
        )
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
    priors = asof_prior_grids(s, retrospective=True)
    honest = priors.index[~priors["retrospective"].astype(bool)]
    history = wide.dropna(subset=[B.FINAL_N]).index.intersection(honest)
    zcal = calibration(wide, priors, meas, history)
    grid = priors[QCOLS].to_numpy()
    m0, s0 = B.prior_from_grid(grid)

    out = pd.DataFrame(index=priors.index)
    out["player_name"] = out.index.map(players["player_name"])
    out["draft_year"] = out.index.map(players["draft_year"]).astype(int)
    out["pick"] = out.index.map(players["pick_overall"])
    out["drafted_by"] = out.index.map(players["team_id"])
    out["team"] = out.index.map(players["rights_team"])  # after draft-night trades
    out["franchise"] = out["team"].replace(FRANCHISE)
    out["prospect_source"] = out.index.map(players["prospect_source"])
    out["projection_type"] = np.where(
        priors["retrospective"].astype(bool), "retrospective", "as of draft night"
    )
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
    # Peak value through year 8 can never be below the peak already reached.
    reached = np.nan_to_num(out["peak_so_far"].to_numpy(dtype=float), nan=0.0)
    for n in range(1, B.FINAL_N):
        rows = (seasons == n) & ~final
        if rows.any():
            q = zcal[n].quantiles(B.Posterior(mean[rows], sd[rows], weight[rows]))
            q = np.maximum(q, reached[rows, None])
            pq[rows] = B.shrink_floor(q, reached[rows], n)
    pq[final] = (B.s(obs_final[final]) ** 2)[:, None]
    pq = np.maximum(pq, reached[:, None])

    out["projected_floor"] = grid[:, M.qidx(M.FLOOR)]
    out["projected_median"] = grid[:, M.qidx(M.MEDIAN)]
    out["projected_ceiling"] = grid[:, M.qidx(M.CEILING)]
    out["current_floor"] = pq[:, M.qidx(M.FLOOR)]
    out["current_median"] = pq[:, M.qidx(M.MEDIAN)]
    out["current_ceiling"] = pq[:, M.qidx(M.CEILING)]
    out["data_weight"] = weight
    # Plain-language summaries: the tier the median falls in, and P(All-Star or better).
    cuts = tier_cuts(s)
    star = TIERS.index("All-Star")
    for kind, q in (("projected", grid), ("current", pq)):
        med = q[:, M.qidx(M.MEDIAN)]
        out[f"{kind}_tier"] = [TIERS[i] for i in np.searchsorted(cuts, med, side="right")]
        tp = tier_probabilities(q, cuts)
        out[f"{kind}_p_all_star"] = tp[:, star:].sum(axis=1)
        # Bust risk: chance the career never reaches rotation level (Out of league or Bust).
        out[f"{kind}_p_bust"] = tp[:, : TIERS.index("Rotation")].sum(axis=1)
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
        if n == 0:
            cal = grid
        else:
            cal = np.maximum(zcal[n].quantiles(post), obs[:, None])
            cal = B.shrink_floor(cal, obs, n)
        variants.append(("calibrated", cal))
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
