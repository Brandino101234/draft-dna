import numpy as np
import pandas as pd
import pytest

from draft_dna.grading import bayes as B


def test_data_weight_grows_with_information() -> None:
    m0, s0 = np.array([1.0]), np.array([0.5])
    weak = B.update(m0, s0, np.array([1.0]), B.Measurement(1, 0.0, 0.25, 0.15))
    strong = B.update(m0, s0, np.array([1.0]), B.Measurement(5, 0.0, 0.9, 0.12))
    assert 0 < weak.data_weight[0] < strong.data_weight[0] < 1
    assert strong.sd[0] < weak.sd[0] < s0[0]


def test_posterior_mean_matches_closed_form() -> None:
    m0, s0 = np.array([1.0]), np.array([0.5])
    meas = B.Measurement(3, 0.1, 0.7, 0.2)
    obs_peak = np.array([1.44])  # sqrt = 1.2
    post = B.update(m0, s0, obs_peak, meas)
    prec = 1 / 0.25 + 0.7**2 / 0.04
    expected = (1.0 / 0.25 + 0.7 * (1.2 - 0.1) / 0.04) / prec
    assert post.mean[0] == pytest.approx(expected)
    assert post.sd[0] == pytest.approx(np.sqrt(1 / prec))


def test_no_seasons_returns_prior() -> None:
    post = B.update(np.array([1.2]), np.array([0.4]), np.array([np.nan]), None)
    assert post.mean[0] == 1.2 and post.data_weight[0] == 0


def test_measurement_model_recovers_simulated_relationship() -> None:
    rng = np.random.default_rng(0)
    final = rng.gamma(1.5, 0.8, 3000)
    obs = (0.05 + 0.8 * np.sqrt(final) + rng.normal(0, 0.1, 3000)) ** 2
    peaks = pd.DataFrame({4: obs, 8: final, 1: obs, 2: obs, 3: obs, 5: obs, 6: obs, 7: obs})
    m = B.fit_measurement(peaks)[4]
    assert m.b == pytest.approx(0.8, abs=0.03) and m.sigma == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize(
    ("n", "retired", "expected"),
    [
        (0, False, "Projection"),
        (1, False, "Provisional (low confidence)"),
        (3, False, "Provisional (medium confidence)"),
        (4, False, "Year-4 Verdict"),
        (8, False, "Career Grade"),
        (2, True, "Career Grade (retired)"),
    ],
)
def test_status_labels(n: int, retired: bool, expected: str) -> None:
    assert B.status(n, retired) == expected


def test_plays_like_picks_nearest_style_within_height_and_never_self() -> None:
    import pandas as pd

    from draft_dna.grading.extras import plays_like

    nba = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05], [0.0, 1.0]],
        index=["self", "tall_twin", "near", "far"],
    )
    heights = pd.Series({"self": 78.0, "tall_twin": 85.0, "near": 79.0, "far": 77.0})
    out = plays_like(nba, nba, heights, pd.Index(["self"]), use_nba_for=nba.index)
    ids = out["plays_like_id"].tolist()
    assert "self" not in ids and "tall_twin" not in ids  # 7 inches taller is filtered out
    assert ids[0] == "near" and out["basis"].eq("NBA shots").all()


def test_recruit_groups_treat_missing_rank_as_unranked() -> None:
    import pandas as pd

    from draft_dna.eval.recruits import recruit_group

    g = recruit_group(pd.Series([1, 10, 11, 25, 26, 50, 51, 100, None]))
    assert g.tolist() == [
        "RSCI 1-10",
        "RSCI 1-10",
        "RSCI 11-25",
        "RSCI 11-25",
        "RSCI 26-50",
        "RSCI 26-50",
        "RSCI 51-100",
        "RSCI 51-100",
        "Unranked",
    ]
