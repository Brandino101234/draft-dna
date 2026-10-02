"""Rolling-origin backtest by draft year, with leakage guards.

Target: a player's peak value (best 3-season blend value) through season H = 6 after
the draft. To predict draft class Y, a model may learn only from classes whose H-season
outcome was complete by draft night of Y: class c qualifies when c + H <= Y.

Each fold refits every scaler, imputer and model on its own training rows. The fold
object records what it was trained on so tests can assert nothing leaked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval import metrics as M
from draft_dna.ingest.storage import read_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

HORIZON = 6
TARGET = "y_peak6"
FIRST_TEST_YEAR = 2006


class QuantileModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> QuantileModel: ...

    def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_years: tuple[int, ...]

    @property
    def max_train_year(self) -> int:
        return max(self.train_years)


def modeling_frame(s: Settings) -> pd.DataFrame:
    """Drafted players with pre-draft features and (where observed) the H-season target."""
    f = read_table("modeled", "features", "predraft", s)
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    at_h = otn[otn["n"] == HORIZON].set_index("bbref_id")
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    df = f[f["drafted"]].set_index("bbref_id")
    df["player_name"] = players["player_name"]
    df[TARGET] = at_h["peak3_blend"]
    df["composite6"] = at_h["composite_blend"]
    return df.reset_index()


def folds(
    df: pd.DataFrame,
    first: int = FIRST_TEST_YEAR,
    last: int | None = None,
    target: str = TARGET,
    lag: int = HORIZON,
) -> Iterator[Fold]:
    """`lag` = how many years before Y a class must be to train year Y. The default
    (= horizon) is the strict rule: its outcome must be known by draft night."""
    observed = df.loc[df[target].notna(), "draft_year"]
    last = last or int(observed.max())
    for y in range(first, last + 1):
        train = tuple(sorted(int(c) for c in observed.unique() if c + lag <= y))
        if train:
            yield Fold(y, train)


def split(
    df: pd.DataFrame, fold: Fold, target: str = TARGET, lag: int = HORIZON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["draft_year"].isin(fold.train_years) & df[target].notna()]
    test = df[(df["draft_year"] == fold.test_year) & df[target].notna()]
    assert train["draft_year"].max() + lag <= fold.test_year, "leakage: outcome not yet observed"
    assert not set(train["bbref_id"]) & set(test["bbref_id"]), "leakage: player in both sets"
    return train, test


def run_backtest(
    df: pd.DataFrame,
    make_model: Callable[[], QuantileModel],
    first: int = FIRST_TEST_YEAR,
    last: int | None = None,
    target: str = TARGET,
    lag: int = HORIZON,
) -> pd.DataFrame:
    """Out-of-sample quantile predictions for every test player, one fold per year."""
    rows = []
    for fold in folds(df, first, last, target, lag):
        train, test = split(df, fold, target, lag)
        model = make_model().fit(train, train[target].to_numpy())
        q = M.monotone(model.predict_quantiles(test))
        out = pd.DataFrame(q, columns=[f"q{t:.2f}" for t in M.QS])
        out.insert(0, "bbref_id", test["bbref_id"].to_numpy())
        out.insert(1, "draft_year", fold.test_year)
        out.insert(2, "y", test[target].to_numpy())
        out.insert(3, "n_train", len(train))
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def qmatrix(pred: pd.DataFrame) -> np.ndarray:
    return pred[[f"q{t:.2f}" for t in M.QS]].to_numpy()
