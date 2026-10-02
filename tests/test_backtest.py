import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from draft_dna.eval import backtest as bt
from draft_dna.eval import metrics as M
from draft_dna.models.knn import PickBaseline


def test_pinball_penalizes_asymmetrically() -> None:
    y = np.array([1.0])
    # A 90th-percentile prediction that is too low costs 9x one that is too high.
    low = M.pinball(y, np.array([0.0]), 0.9)[0]
    high = M.pinball(y, np.array([2.0]), 0.9)[0]
    assert low == pytest.approx(9 * high)


def test_crps_from_quantile_grid_matches_normal_closed_form() -> None:
    # Closed-form CRPS of N(0,1) at y=0.3, vs the 19-quantile approximation.
    y = 0.3
    exact = y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / np.sqrt(np.pi)
    q = norm.ppf(M.QS)[None, :]
    approx = M.crps(np.array([y]), q)[0]
    assert approx == pytest.approx(exact, rel=0.12)  # coarse grid, slight underestimate


def test_cdf_inverts_quantiles_including_point_mass() -> None:
    q = norm.ppf(M.QS)[None, :]
    assert M.cdf_at(q, 0.0)[0] == pytest.approx(0.5, abs=1e-6)
    # 40% of mass at exactly zero, then spread above.
    q2 = np.where(M.QS <= 0.4, 0.0, M.QS - 0.4)[None, :]
    assert M.cdf_at(q2, 0.0)[0] == pytest.approx(0.4)
    assert M.cdf_at(q2, -1.0)[0] == 0.0 and M.cdf_at(q2, 5.0)[0] == 1.0


def test_coverage_counts_ties_as_half() -> None:
    assert M.coverage_below(np.array([0.0, 0.0, 1.0, -1.0]), np.zeros(4)) == pytest.approx(0.5)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(1996, 2021), 20)
    return pd.DataFrame(
        {
            "bbref_id": [f"p{i}" for i in range(len(years))],
            "draft_year": years,
            "pick": np.tile(np.arange(1, 21), 25),
            bt.TARGET: rng.gamma(1.0, 0.5, len(years)),
        }
    )


def test_folds_never_train_on_unobserved_outcomes() -> None:
    df = _frame()
    for fold in bt.folds(df):
        train, test = bt.split(df, fold)
        assert train["draft_year"].max() + bt.HORIZON <= fold.test_year
        assert (test["draft_year"] == fold.test_year).all()
        assert not set(train.bbref_id) & set(test.bbref_id)


def test_first_fold_trains_only_on_classes_six_years_back() -> None:
    first = next(bt.folds(_frame()))
    assert first.test_year == bt.FIRST_TEST_YEAR
    assert max(first.train_years) == bt.FIRST_TEST_YEAR - bt.HORIZON


def test_split_rejects_leaky_fold() -> None:
    df = _frame()
    bad = bt.Fold(2010, tuple(range(1996, 2009)))
    with pytest.raises(AssertionError, match="leakage"):
        bt.split(df, bad)


def test_backtest_outputs_monotone_quantiles_for_every_test_player() -> None:
    df = _frame()
    pred = bt.run_backtest(df, lambda: PickBaseline(k=20))
    q = bt.qmatrix(pred)
    assert (np.diff(q, axis=1) >= 0).all()
    assert len(pred) == (df.draft_year >= bt.FIRST_TEST_YEAR).sum()
