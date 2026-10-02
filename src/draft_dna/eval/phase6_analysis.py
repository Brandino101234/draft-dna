"""Phase 6 analyses on the player outcomes-vs-projection table.

- segment_rates: beat-ceiling / below-floor rates and mean PIT by group, with CIs
- team_effects: drafting-franchise effects with partial pooling (empirical Bayes)
- situation_effects: propensity-weighted associations with balance checks and E-values
- shap_overperformance: can pre-draft traits predict beating the projection? (+ SHAP)
- verdict_reversals: Year-4 Verdict vs Career Grade
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

Z = norm.ppf(0.975)


def wilson(k: float, n: float) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def segment_rates(df: pd.DataFrame, by: str, n: int = 6) -> pd.DataFrame:
    d = df[df[f"pit{n}"].notna()]
    rows = []
    for g, x in d.groupby(by, observed=True):
        k_up = (x[f"verdict{n}"] == "beat ceiling").sum()
        k_dn = (x[f"verdict{n}"] == "below floor").sum()
        lo_u, hi_u = wilson(k_up, len(x))
        lo_d, hi_d = wilson(k_dn, len(x))
        se = x[f"pit{n}"].std() / np.sqrt(len(x))
        rows.append(
            {
                by: g,
                "n": len(x),
                "beat_ceiling": k_up / len(x),
                "bc_lo": lo_u,
                "bc_hi": hi_u,
                "below_floor": k_dn / len(x),
                "bf_lo": lo_d,
                "bf_hi": hi_d,
                "mean_pit": x[f"pit{n}"].mean(),
                "pit_lo": x[f"pit{n}"].mean() - Z * se,
                "pit_hi": x[f"pit{n}"].mean() + Z * se,
            }
        )
    return pd.DataFrame(rows)


def team_effects(df: pd.DataFrame, n: int = 6) -> tuple[pd.DataFrame, dict[str, float]]:
    """Each franchise's mean PIT with its own CI, a heterogeneity test (do franchises
    differ more than luck would produce?), and partially pooled estimates.

    Cochran's Q compares the spread of team means to what sampling noise alone produces;
    under "no real team differences" Q ~ chi-square with (teams - 1) degrees of freedom.
    The pooled estimate shrinks each team toward the league mean by tau^2/(tau^2+noise),
    where tau^2 is the between-team variance left after subtracting noise (DerSimonian-Laird).
    """
    from scipy.stats import chi2

    d = df[df[f"pit{n}"].notna()]
    g = d.groupby("franchise")[f"pit{n}"].agg(["mean", "count"])
    sigma2 = d[f"pit{n}"].var()
    se2 = sigma2 / g["count"]
    w = 1 / se2
    grand = float(np.sum(w * g["mean"]) / np.sum(w))
    q_stat = float(np.sum(w * (g["mean"] - grand) ** 2))
    k = len(g)
    p_value = float(chi2.sf(q_stat, k - 1))
    tau2 = max((q_stat - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)), 0.0)
    shrink = tau2 / (tau2 + se2)
    g["raw_lo"] = g["mean"] - Z * np.sqrt(se2)
    g["raw_hi"] = g["mean"] + Z * np.sqrt(se2)
    g["pooled"] = grand + shrink * (g["mean"] - grand)
    g = g.rename(columns={"mean": "raw_mean", "count": "n"}).sort_values("raw_mean")
    stats = {
        "teams": k,
        "Q": q_stat,
        "df": k - 1,
        "p_value": p_value,
        "tau": float(np.sqrt(tau2)),
        "league_mean": grand,
    }
    return g, stats


SITUATIONS = {
    # name: (column, population, label, timing)
    "weak_team": ("weak_team", "all", "Drafted by a bottom-third team (by SRS)", "pre-draft"),
    "crowded_position": (
        "crowded_position",
        "all",
        "Drafting team deep at his position (top half of minutes share)",
        "pre-draft",
    ),
    "coach_change": (
        "coach_change",
        "all",
        "Drafting team changed head coach in years 1-3",
        "post-draft, team-level",
    ),
    "traded_early": (
        "traded_early",
        "played",
        "Traded / moved teams in years 1-3",
        "post-draft, player-level",
    ),
    "low_availability": (
        "low_availability",
        "played",
        "Missed >= 25% of games in years 1-2 (injury, G League or DNP)",
        "post-draft, player-level",
    ),
}
PS_FEATURES = [
    "log_pick",
    "age_at_draft",
    "height_in",
    "pts_per40",
    "trb_per40",
    "ast_per40",
    "ts_rel",
    "usg_pct",
    "bpm",
    "draft_year",
    "src_college",
    "src_high_school",
    "src_other_team",
    "pos_guard",
    "pos_forward",
    "pos_big",
]


def e_value(rr: float) -> float:
    """VanderWeele & Ding: how strongly an unmeasured confounder would need to be
    associated with both treatment and outcome (risk-ratio scale) to explain away rr."""
    if not np.isfinite(rr) or rr <= 0:
        return np.nan
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def _design(df: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = feats.reindex(df["bbref_id"]).reset_index(drop=True)
    x["log_pick"] = np.log(x["pick"].astype(float))
    x = x[PS_FEATURES].astype(float)
    flags = x[["pts_per40", "bpm"]].isna().astype(float).add_prefix("missing_")
    x = x.fillna(x.median())
    return pd.concat([x, flags], axis=1), np.ones(len(x))


def _ipw(x: pd.DataFrame, t: np.ndarray, y: np.ndarray, hit: np.ndarray) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit_transform(x)
    e = LogisticRegression(C=1.0, max_iter=2000).fit(xs, t).predict_proba(xs)[:, 1]
    keep = (e > 0.05) & (e < 0.95)  # trim regions without overlap
    t, y, hit, e = t[keep], y[keep], hit[keep], e[keep]
    w = np.where(t == 1, t.mean() / e, (1 - t.mean()) / (1 - e))  # stabilized weights
    m1, m0 = np.average(y[t == 1], weights=w[t == 1]), np.average(y[t == 0], weights=w[t == 0])
    h1, h0 = np.average(hit[t == 1], weights=w[t == 1]), np.average(hit[t == 0], weights=w[t == 0])
    return {
        "pit_diff": m1 - m0,
        "rr_above_median": h1 / h0 if h0 > 0 else np.nan,
        "n_used": int(keep.sum()),
        "e_min": float(e.min()),
        "e_max": float(e.max()),
    }


def smd(x: pd.DataFrame, t: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    """Standardized mean difference of each covariate between groups (|SMD| < 0.1 = balanced)."""
    w = np.ones(len(t)) if w is None else w
    out = {}
    for c in x.columns:
        v = x[c].to_numpy()
        m1 = np.average(v[t == 1], weights=w[t == 1])
        m0 = np.average(v[t == 0], weights=w[t == 0])
        sd = np.sqrt((v[t == 1].var() + v[t == 0].var()) / 2) or 1.0
        out[c] = (m1 - m0) / sd
    return pd.Series(out)


def situation_effects(
    df: pd.DataFrame, feats: pd.DataFrame, n: int = 6, n_boot: int = 300, seed: int = 0
) -> pd.DataFrame:
    """Raw and propensity-weighted differences in PIT (and risk ratio of beating the
    projected median) for each situation, with bootstrap CIs, balance and E-values."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows = []
    for col, pop, label, timing in SITUATIONS.values():
        d = df[df[f"pit{n}"].notna() & df[col].notna()]
        if pop == "played":
            d = d[d["played_y1_3"].astype(bool)]
        d = d.reset_index(drop=True)
        x, _ = _design(d, feats)
        t = d[col].astype(bool).to_numpy().astype(int)
        y = d[f"pit{n}"].to_numpy()
        hit = (y > 0.5).astype(float)
        est = _ipw(x, t, y, hit)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(d), len(d))
            if t[i].min() == t[i].max():
                continue
            boots.append(_ipw(x.iloc[i].reset_index(drop=True), t[i], y[i], hit[i]))
        b = pd.DataFrame(boots)
        xs = StandardScaler().fit_transform(x)
        e = LogisticRegression(C=1.0, max_iter=2000).fit(xs, t).predict_proba(xs)[:, 1]
        w = np.where(t == 1, t.mean() / e, (1 - t.mean()) / (1 - e))
        rr_lo, rr_hi = np.nanpercentile(b["rr_above_median"], [2.5, 97.5])
        ci_bound = rr_lo if est["rr_above_median"] >= 1 else rr_hi
        rows.append(
            {
                "situation": label,
                "timing": timing,
                "n": len(d),
                "treated_share": t.mean(),
                "raw_pit_diff": y[t == 1].mean() - y[t == 0].mean(),
                "ipw_pit_diff": est["pit_diff"],
                "ipw_lo": np.percentile(b["pit_diff"], 2.5),
                "ipw_hi": np.percentile(b["pit_diff"], 97.5),
                "rr_above_median": est["rr_above_median"],
                "rr_lo": rr_lo,
                "rr_hi": rr_hi,
                "e_value": e_value(est["rr_above_median"]),
                "e_value_ci": 1.0 if rr_lo <= 1 <= rr_hi else e_value(ci_bound),
                "max_abs_smd_before": smd(x, t).abs().max(),
                "max_abs_smd_after": smd(x, t, w).abs().max(),
            }
        )
    return pd.DataFrame(rows)


def shap_overperformance(
    df: pd.DataFrame, feats: pd.DataFrame, n: int = 6, seed: int = 0
) -> dict[str, object]:
    """Can pre-draft traits predict *beating the projection* (PIT)? Then explain with SHAP.

    The projection already uses draft slot, so PIT measures surprise relative to slot.
    Predictive power is checked first with leave-draft-classes-out cross-validation; SHAP
    is only meaningful if the model predicts out of sample.
    """
    import lightgbm as lgb
    import shap
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    from draft_dna.features.predraft import STATS_FEATURES
    from draft_dna.features.shot_dna import SHOT_FEATURES

    d = df[df[f"pit{n}"].notna()].reset_index(drop=True)
    x = feats.reindex(d["bbref_id"]).reset_index(drop=True)
    shot = [c for c in SHOT_FEATURES if c in x]
    cols = [c for c in STATS_FEATURES + shot if c in x]
    x = x[cols].astype(float)
    x["log_pick"] = np.log(d["pick"].astype(float))
    y = d[f"pit{n}"].to_numpy()
    params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": seed,
    }
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups=d["draft_year"]):
        oof[te] = lgb.LGBMRegressor(**params).fit(x.iloc[tr], y[tr]).predict(x.iloc[te])
    rho = spearmanr(oof, y).statistic
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(seed)
    null = [spearmanr(oof, rng.permutation(y)).statistic for _ in range(500)]
    model = lgb.LGBMRegressor(**params).fit(x, y)
    sv = shap.TreeExplainer(model).shap_values(x)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=x.columns).sort_values(ascending=False)
    direction = pd.Series(
        [spearmanr(x[c], sv[:, i], nan_policy="omit").statistic for i, c in enumerate(x.columns)],
        index=x.columns,
    )
    return {
        "oof_spearman": float(rho),
        "oof_r2": float(r2),
        "null_p": float(np.mean(np.abs(null) >= abs(rho))),
        "importance": importance,
        "direction": direction,
        "shap_values": sv,
        "x": x,
    }


def rolling_overperformance(
    df: pd.DataFrame, feats: pd.DataFrame, n: int = 6, first: int = 2010, seed: int = 0
) -> pd.DataFrame:
    """Time-respecting check: predict each class's PIT using only classes whose outcome
    was known by its draft night (c + n <= Y); report rank correlation per year and pooled."""
    import lightgbm as lgb
    from scipy.stats import spearmanr

    from draft_dna.features.predraft import STATS_FEATURES

    d = df[df[f"pit{n}"].notna()].reset_index(drop=True)
    x = feats.reindex(d["bbref_id"]).reset_index(drop=True)
    x = x[[c for c in STATS_FEATURES if c in x]].astype(float)
    x["log_pick"] = np.log(d["pick"].astype(float))
    y = d[f"pit{n}"].to_numpy()
    params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": seed,
    }
    pred = pd.Series(np.nan, index=d.index)
    for year in sorted(d["draft_year"].unique()):
        if year < first:
            continue
        tr = d["draft_year"] + n <= year
        te = d["draft_year"] == year
        if tr.sum() < 150:
            continue
        pred[te] = lgb.LGBMRegressor(**params).fit(x[tr], y[tr]).predict(x[te])
    ok = pred.notna()
    rows = [
        {"draft_year": int(yr), "n": int(g.sum()), "spearman": spearmanr(pred[g], y[g]).statistic}
        for yr, g in (
            (yr, ok & (d["draft_year"] == yr)) for yr in sorted(d.loc[ok, "draft_year"].unique())
        )
    ]
    out = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    pooled = spearmanr(pred[ok], y[ok]).statistic
    null = [spearmanr(pred[ok], rng.permutation(y[ok])).statistic for _ in range(1000)]
    out.attrs["pooled_spearman"] = float(pooled)
    out.attrs["perm_p"] = float(np.mean(np.abs(null) >= abs(pooled)))
    out.attrs["n"] = int(ok.sum())
    return out
