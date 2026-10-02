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


def test_tier_probabilities_sum_to_one_and_follow_cdf() -> None:
    from draft_dna.models.projections import tier_probabilities

    q = np.vstack([norm.ppf(M.QS, loc=1.0, scale=0.8), np.linspace(0, 3, len(M.QS))])
    cuts = [0.01, 0.32, 1.01, 1.76, 2.37]
    tp = tier_probabilities(q, cuts)
    assert np.allclose(tp.sum(axis=1), 1.0)
    assert (tp >= 0).all()
    # P(All-Star or better) equals 1 - CDF at the All-Star cutoff.
    assert tp[0, 4:].sum() == pytest.approx(1 - M.cdf_at(q[:1], 1.76)[0])


def test_conformal_shifts_quantiles_to_match_calibration_errors() -> None:
    from draft_dna.models.distribution import Conformal

    class Constant:
        name = "constant"

        def fit(self, train: pd.DataFrame, y: np.ndarray) -> "Constant":
            return self

        def predict_quantiles(self, test: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(test), len(M.QS)))  # always predicts 0 everywhere

    rng = np.random.default_rng(1)
    years = np.repeat(np.arange(2000, 2010), 200)
    train = pd.DataFrame({"draft_year": years})
    y = rng.uniform(0, 1, len(years))  # true quantile tau = tau
    model = Conformal(Constant, calib_classes=3, min_fit_classes=4).fit(train, y)
    q = model.predict_quantiles(pd.DataFrame({"draft_year": [2010]}))[0]
    assert np.allclose(q, M.QS, atol=0.05)


def test_stats_knn_does_not_pick_hub_players_with_missing_data() -> None:
    from draft_dna.models.knn import StatsKnn

    feats = ["a", "b", "c", "d"]
    train = pd.DataFrame(
        {
            "bbref_id": ["full_near", "full_far", "hub"],
            "a": [1.0, 5.0, np.nan],
            "b": [1.0, 5.0, np.nan],
            "c": [1.0, 5.0, np.nan],
            "d": [1.0, 5.0, 1.2],
        }
    )
    test = pd.DataFrame({"bbref_id": ["x"], "a": [1.1], "b": [0.9], "c": [1.0], "d": [1.2]})
    m = StatsKnn(k=2, features=feats).fit(train, np.zeros(3))
    idx, _ = m.neighbors(test)
    # Without the overlap guard, "hub" (only feature d, identical) would rank first.
    assert train.iloc[idx[0][0]]["bbref_id"] == "full_near"
    assert "hub" not in set(train.iloc[idx[0]]["bbref_id"])


def test_guardrail_blocks_comps_of_different_size_or_position() -> None:
    from draft_dna.models.knn import GuardedKnn

    feats = ["rim_rate"]
    train = pd.DataFrame(
        {
            "bbref_id": ["big", "guard_far", "guard_near"],
            "rim_rate": [0.60, 0.20, 0.55],
            "height_in": [83.0, 74.0, 75.0],
            "position": ["big", "guard", "guard"],
        }
    )
    test = pd.DataFrame(
        {"bbref_id": ["g"], "rim_rate": [0.60], "height_in": [74.5], "position": ["guard"]}
    )
    m = GuardedKnn(k=2, features=feats).fit(train, np.zeros(3))
    idx, _ = m.neighbors(test)
    # Identical shot profile, but the 6'11" big is not an eligible comp for a 6'2" guard.
    assert list(train.iloc[idx[0]]["bbref_id"]) == ["guard_near", "guard_far"]


def test_spatial_model_fills_style_weights_only_for_players_with_maps() -> None:
    from draft_dna.eval.phase5 import STYLE_K, SpatialLgbm
    from draft_dna.features import shot_xy

    rng = np.random.default_rng(0)
    n_cells = len(shot_xy.GRID_X) * len(shot_xy.GRID_Y)
    maps = pd.DataFrame(
        rng.random((30, n_cells)),
        index=[f"p{i}" for i in range(30)],
        columns=[f"c{i}" for i in range(n_cells)],
    )
    m = SpatialLgbm(maps)
    m.nmf = shot_xy.fit_styles(maps, k=STYLE_K)
    df = pd.DataFrame({"bbref_id": ["p0", "p1", "no_map"]})
    f = m._features(df)
    styles = f[[f"style_{i}" for i in range(STYLE_K)]]
    assert styles.iloc[:2].notna().all().all()  # players with maps get weights
    assert styles.iloc[2].isna().all()  # and players without a map stay missing
    assert np.allclose(styles.iloc[:2].sum(axis=1), 1.0)
