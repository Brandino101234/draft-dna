"""Distributional models that output a quantile grid for each prospect.

- LgbmQuantile:  one gradient-boosted model per quantile (pinball objective).
- NgbQuantile:   NGBoost with a Normal distribution on a transformed target.
- BayesTobit:    Bayesian hierarchical regression (PyMC) on the transformed target,
                 censored at zero, with partially pooled intercepts by position, era and
                 pick band.
- Conformal:     wraps any model; shifts each predicted quantile by the error it made on
                 the most recent held-out classes, so coverage matches its level.

Target transform: t(y) = log(1 + max(y, 0) / 0.25). It compresses the long right tail
(superstars) and maps "no NBA value" to exactly 0, the censoring point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import lightgbm as lgb
import numpy as np
import pandas as pd

from draft_dna.eval.metrics import QS
from draft_dna.features.predraft import STATS_FEATURES

SCALE = 0.25


def to_t(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(y, 0.0) / SCALE)


def from_t(t: np.ndarray) -> np.ndarray:
    return SCALE * np.expm1(np.maximum(t, 0.0))


def to_sqrt(y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(y, 0.0))


def from_sqrt(t: np.ndarray) -> np.ndarray:
    return np.maximum(t, 0.0) ** 2


TRANSFORMS = {"log": (to_t, from_t), "sqrt": (to_sqrt, from_sqrt)}


def with_pick(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_pick"] = np.log(out["pick"].astype(float))
    return out


PICK_STATS = ["log_pick", *STATS_FEATURES]


class _Imputer:
    """Median imputation + missingness flags for feature groups (fit inside the fold)."""

    GROUP_FLAGS: ClassVar[dict[str, str]] = {
        "no_combine": "wingspan_in", "no_college": "pts_per40", "no_bpm": "bpm",
        "no_athletic": "max_vertical",
    }  # fmt: skip

    MIN_OBSERVED = 30  # a feature needs this many non-missing training values to be used
    CLIP = 5.0

    def fit(self, x: pd.DataFrame) -> _Imputer:
        x = x.astype(float)
        self.cols = list(x.columns)
        # Features (almost) absent from this fold's training data, e.g. BPM before 2008,
        # are neutralized: otherwise a test player's raw value enters unscaled.
        self.unused = [c for c in self.cols if x[c].notna().sum() < self.MIN_OBSERVED]
        self.med = x.median().fillna(0.0)
        self.mu = x.fillna(self.med).mean()
        self.sd = x.fillna(self.med).std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, x: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
        out = x[self.cols].astype(float).fillna(self.med)
        out[self.unused] = self.med[self.unused]
        if standardize:
            out = ((out - self.mu) / self.sd).clip(-self.CLIP, self.CLIP)
        for flag, col in self.GROUP_FLAGS.items():
            if col in x:
                out[flag] = x[col].isna().astype(float)
        return out.fillna(0.0)


class LgbmQuantile:
    name = "LightGBM quantile"

    def __init__(
        self,
        features: list[str] | None = None,
        label: str | None = None,
        n_estimators: int = 250,
        min_child_samples: int = 25,
        num_leaves: int = 7,
    ) -> None:
        self.features = features or PICK_STATS
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "num_leaves": num_leaves,
        }
        if label:
            self.name = label

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> LgbmQuantile:
        x = with_pick(train)[self.features]
        self.models = [
            lgb.LGBMRegressor(
                objective="quantile", alpha=float(tau), learning_rate=0.03, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                random_state=0, **self.params,
            ).fit(x, y)
            for tau in QS
        ]  # fmt: skip
        return self

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        x = with_pick(test)[self.features]
        return np.column_stack([m.predict(x) for m in self.models])


class NgbQuantile:
    name = "NGBoost"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features or PICK_STATS

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> NgbQuantile:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        x = with_pick(train)[self.features]
        self.imp = _Imputer().fit(x)
        self.model = NGBRegressor(
            Dist=Normal, n_estimators=300, learning_rate=0.03, verbose=False, random_state=0
        ).fit(self.imp.transform(x).to_numpy(dtype=float), to_t(y))
        return self

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        from scipy.stats import norm

        x = self.imp.transform(with_pick(test)[self.features]).to_numpy(dtype=float)
        dist = self.model.pred_dist(x)
        mu, sd = dist.params["loc"], dist.params["scale"]
        qt = norm.ppf(QS)[None, :] * sd[:, None] + mu[:, None]
        return from_t(qt)  # mass below 0 collapses to "no NBA value"


ERA_EDGES = [1996, 2002, 2008, 2014, 9999]
PICK_EDGES = [0, 5, 14, 30, 99]
BAYES_FEATURES = ["log_pick", "age_at_draft", "height_in", "wingspan_minus_height", "pts_per40",
                  "trb_per40", "ast_per40", "stl_per40", "blk_per40", "tov_per40", "ts_rel",
                  "ftr", "fg3a_rate", "ft_pct", "usg_pct", "bpm", "team_sos", "college_seasons",
                  "recruit_rank_top100", "src_high_school", "src_other_team"]  # fmt: skip


POSITION_CODE = {"guard": 0, "forward": 1, "big": 2}


def _groups(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "position": df["position"].map(POSITION_CODE).fillna(3).astype(int).to_numpy(),
        "era": np.digitize(df["draft_year"], ERA_EDGES[1:-1]),
        "pick_band": np.digitize(df["pick"], PICK_EDGES[1:-1], right=True),
    }  # fmt: skip


GROUP_SIZES = {"position": 4, "era": len(ERA_EDGES) - 1, "pick_band": len(PICK_EDGES) - 1}


class BayesTobit:
    """y*_i ~ Normal(mu_i, sigma); observed t(y_i) = max(y*_i, 0)  (censored at zero)
    mu_i = a + X_i beta + u_position + u_era + u_pickband
    u_g ~ Normal(0, tau_g), tau_g ~ HalfNormal(0.5)   (partial pooling)
    beta ~ Normal(0, 0.3)                            (shrinks weak predictors to 0)
    """

    name = "Bayesian hierarchical (Tobit)"

    def __init__(
        self,
        draws: int = 500,
        tune: int = 500,
        seed: int = 0,
        transform: str = "log",
        sigma_by_pick: bool = False,
    ) -> None:
        self.draws, self.tune, self.seed = draws, tune, seed
        self.transform, self.sigma_by_pick = transform, sigma_by_pick
        self.fwd, self.inv = TRANSFORMS[transform]

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> BayesTobit:
        import pymc as pm

        x = with_pick(train)[BAYES_FEATURES]
        self.imp = _Imputer().fit(x)
        xs = self.imp.transform(x).to_numpy(dtype=float)
        self.cols = list(self.imp.transform(x.head(1)).columns)
        g = _groups(train)
        with pm.Model():
            a = pm.Normal("a", 0.0, 1.0)
            beta = pm.Normal("beta", 0.0, 0.3, shape=xs.shape[1])
            mu = a + pm.math.dot(xs, beta)
            for name, size in GROUP_SIZES.items():
                tau = pm.HalfNormal(f"tau_{name}", 0.5)
                z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=size)  # non-centered
                u = pm.Deterministic(f"u_{name}", z * tau)
                mu = mu + u[g[name]]
            if self.sigma_by_pick:
                sigma_g = pm.HalfNormal("sigma", 1.0, shape=GROUP_SIZES["pick_band"])
                sigma = sigma_g[g["pick_band"]]
            else:
                sigma = pm.HalfNormal("sigma", 1.0)
            pm.Censored("obs", pm.Normal.dist(mu, sigma), lower=0.0, upper=None,
                        observed=self.fwd(y))  # fmt: skip
            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=2, cores=2, target_accept=0.9,
                random_seed=self.seed, progressbar=False, compute_convergence_checks=False,
            )  # fmt: skip
        return self

    def posterior(self) -> dict[str, np.ndarray]:
        post = self.idata.posterior
        if hasattr(post, "to_dataset"):  # PyMC >= 6 returns an xarray DataTree
            post = post.to_dataset()
        post = post.stack(sample=("chain", "draw"))
        out: dict[str, Any] = {k: post[k].to_numpy() for k in ("a", "beta", "sigma")}
        for name in GROUP_SIZES:
            out[f"u_{name}"] = post[f"u_{name}"].to_numpy()
        return out

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        p = self.posterior()
        xs = self.imp.transform(with_pick(test)[BAYES_FEATURES]).to_numpy(dtype=float)
        g = _groups(test)
        mu = p["a"][None, :] + xs @ p["beta"]
        for name in GROUP_SIZES:
            mu = mu + p[f"u_{name}"][g[name], :]
        rng = np.random.default_rng(self.seed)
        sigma = p["sigma"][g["pick_band"], :] if self.sigma_by_pick else p["sigma"][None, :]
        latent = mu + rng.standard_normal(mu.shape) * sigma
        return np.quantile(self.inv(latent), QS, axis=1).T


class Conformal:
    """Split-conformal quantile calibration (one-sided, per quantile level).

    Inside the fold, the most recent `calib_classes` training classes are held out.
    The base model is fit on the older classes; on the held-out classes we measure
    how far outcomes land from each predicted quantile, and shift that quantile by the
    amount that makes the right share of outcomes fall below it. Recent classes are
    used because the league drifts over time (exchangeability holds best nearby).
    """

    def __init__(self, base: Callable[[], Any], calib_classes: int = 3, min_fit_classes: int = 4):
        self.base, self.k, self.min_fit = base, calib_classes, min_fit_classes
        self.name = f"{base().name} + conformal"

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> Conformal:
        years = sorted(train["draft_year"].unique())
        if len(years) < self.k + self.min_fit:
            self.model = self.base().fit(train, y)
            self.delta = np.zeros(len(QS))
            return self
        calib_mask = train["draft_year"].isin(years[-self.k :]).to_numpy()
        self.model = self.base().fit(train[~calib_mask], y[~calib_mask])
        qc = np.sort(self.model.predict_quantiles(train[calib_mask]), axis=1)
        resid = y[calib_mask][:, None] - qc
        n = len(resid)
        levels = np.clip(np.ceil((n + 1) * QS) / n, 0, 1)
        self.delta = np.array([np.quantile(resid[:, i], lv) for i, lv in enumerate(levels)])
        return self

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        q = np.sort(self.model.predict_quantiles(test), axis=1) + self.delta[None, :]
        return np.maximum(q, 0.0)  # peak value is never negative in practice


class QuantileBlend:
    """Average the quantile grids of several models (quantile averaging / "Vincentization").
    A simple, robust way to combine a strong prior (draft pick) with a stats model."""

    def __init__(self, parts: list[Callable[[], Any]], weights: list[float], name: str) -> None:
        self.parts, self.weights, self.name = parts, np.asarray(weights) / sum(weights), name

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> QuantileBlend:
        self.models = [p().fit(train, y) for p in self.parts]
        return self

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        qs = [np.sort(m.predict_quantiles(test), axis=1) for m in self.models]
        return sum(w * q for w, q in zip(self.weights, qs, strict=True))
