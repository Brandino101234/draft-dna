"""Comp-based (nearest-neighbor) outcome models.

All four share one engine: find each prospect's k most similar historical players and
use the spread of *their* outcomes as the prediction. They differ only in what
"similar" means:

- PickBaseline:        closest draft slots (log scale). Baseline (a): draft pick alone.
- StatsKnn:            standardized pre-draft stats, age and physical profile, every
                       feature weighted equally. Baseline (b).
- SupervisedWeightKnn: the same features, each scaled by how strongly it predicts the
                       outcome in a ridge regression fit on the fold's training rows. A
                       simple learned metric: irrelevant features stop driving comps.
- TreeProximityKnn:    similarity = share of trees in a gradient-boosted model where
                       two players land in the same leaf. Features the model ignores
                       never make two players look alike.

Missing features (no combine, no college) are handled with NaN-aware distances:
distance uses only the features both players have, rescaled for the missing ones.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import nan_euclidean_distances

from draft_dna.eval.metrics import QS
from draft_dna.features.predraft import STATS_FEATURES


class _Standardizer:
    def fit(self, x: pd.DataFrame) -> _Standardizer:
        self.mu = x.mean()
        self.sd = x.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        return ((x - self.mu) / self.sd).to_numpy(dtype=float)


class _NeighborModel:
    name = "knn"
    k = 30

    def __init__(self, k: int | None = None) -> None:
        if k is not None:
            self.k = k

    # Subclasses implement these two.
    def _fit_space(self, train: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def _distances(self, test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> _NeighborModel:
        self.train_ids = train["bbref_id"].to_numpy()
        self.y = np.asarray(y, dtype=float)
        self._fit_space(train, self.y)
        return self

    def neighbors(self, test: pd.DataFrame, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(indices into the training set, distances), nearest first."""
        d = self._distances(test)
        d = np.where(np.isnan(d), np.inf, d)
        k = min(k or self.k, d.shape[1])
        idx = np.argsort(d, axis=1, kind="stable")[:, :k]
        return idx, np.take_along_axis(d, idx, axis=1)

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
        idx, _ = self.neighbors(test)
        return np.quantile(self.y[idx], QS, axis=1).T


class PickBaseline(_NeighborModel):
    name = "Pick only"
    k = 60

    def _fit_space(self, train: pd.DataFrame, y: np.ndarray) -> None:
        self.logpick = np.log(train["pick"].to_numpy(dtype=float))

    def _distances(self, test: pd.DataFrame) -> np.ndarray:
        lp = np.log(test["pick"].to_numpy(dtype=float))
        # Tiny jitter keyed to training order breaks exact-slot ties deterministically.
        return np.abs(lp[:, None] - self.logpick[None, :]) + 1e-9 * np.arange(len(self.logpick))


class StatsKnn(_NeighborModel):
    name = "Stats kNN"

    def __init__(self, k: int | None = None, features: list[str] | None = None) -> None:
        super().__init__(k)
        self.features = features or STATS_FEATURES

    def _weights(self, xs: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.ones(xs.shape[1])

    def _fit_space(self, train: pd.DataFrame, y: np.ndarray) -> None:
        self.scaler = _Standardizer().fit(train[self.features])
        xs = self.scaler.transform(train[self.features])
        self.w = self._weights(xs, y)
        self.xtrain = xs * self.w

    def _distances(self, test: pd.DataFrame) -> np.ndarray:
        xt = self.scaler.transform(test[self.features]) * self.w
        return nan_euclidean_distances(xt, self.xtrain)

    def feature_gaps(self, test_row: pd.DataFrame, train_idx: int) -> pd.Series:
        """Weighted standardized gap per feature between a prospect and one comp
        (small = this feature makes them similar)."""
        a = self.scaler.transform(test_row[self.features])[0] * self.w
        return pd.Series(np.abs(a - self.xtrain[train_idx]), index=self.features)


class SupervisedWeightKnn(StatsKnn):
    name = "Learned-weight kNN"

    def _weights(self, xs: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.where(np.isnan(xs), 0.0, xs)  # standardized: 0 = average
        ridge = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(x, y)
        w = np.abs(ridge.coef_)
        return w / w.sum() * np.sqrt(len(w))  # keep the overall scale comparable


class TreeProximityKnn(_NeighborModel):
    name = "Tree-proximity kNN"

    def __init__(self, k: int | None = None, features: list[str] | None = None) -> None:
        super().__init__(k)
        self.features = features or STATS_FEATURES

    def _fit_space(self, train: pd.DataFrame, y: np.ndarray) -> None:
        self.gbm = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
        ).fit(train[self.features], y)  # fmt: skip
        self.leaves = np.asarray(self.gbm.predict(train[self.features], pred_leaf=True))

    def _distances(self, test: pd.DataFrame) -> np.ndarray:
        lt = np.asarray(self.gbm.predict(test[self.features], pred_leaf=True))
        same = (lt[:, None, :] == self.leaves[None, :, :]).mean(axis=2)
        return 1.0 - same  # 0 = always in the same leaf
