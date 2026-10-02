import numpy as np
import pandas as pd
import pytest

from draft_dna.outcomes.value import (
    _rolling_peak,
    add_composites,
    outcomes_through_n,
    season_grid,
)


def test_rolling_peak_is_best_three_season_average_so_far() -> None:
    v = np.array([1.0, 4.0, 4.0, 4.0, 0.0, 9.0])
    # Early windows are shorter but still divided by 3 (one great rookie season is not
    # a 3-year peak).
    assert _rolling_peak(v).tolist() == pytest.approx([1 / 3, 5 / 3, 3.0, 4.0, 4.0, 13 / 3])


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bbref_id": ["a", "b", "u"],
            "draft_year": [2018.0, 2018.0, np.nan],
            "drafted": [True, True, False],
            "first_season": [2019, pd.NA, 2021],
        }
    ).astype({"first_season": "Int64"})


def _seasons() -> pd.DataFrame:
    rows = [
        # a: plays seasons 1, 2 and 4 (out of the league in season 3)
        ("a", 2019, 2.0), ("a", 2020, 3.0), ("a", 2022, 1.0),
        ("u", 2021, 0.5),
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["bbref_id", "season", "value_vorp"])
    for c in ("value_blend", "value_factor"):
        df[c] = df["value_vorp"]
    for c in ("vorp", "ws", "po_vorp", "po_ws", "po_mp", "mvp_share"):
        df[c] = 0.0
    df["mp"] = 1000.0
    df["games_started"] = 30
    df["games"] = 60
    df["all_star"] = False
    df["all_nba_team"] = None
    return df


def test_grid_zero_fills_missing_seasons_and_counts_from_draft() -> None:
    grid = season_grid(_players(), _seasons(), last_season=2022)
    a = grid[grid.bbref_id == "a"].sort_values("season_num")
    assert a["season_num"].tolist() == [1, 2, 3, 4]
    assert a["value_vorp"].tolist() == [2.0, 3.0, 0.0, 1.0]
    assert a["in_nba"].tolist() == [True, True, False, True]
    # b never played: still present, all zeros (a bust is data, not missing).
    b = grid[grid.bbref_id == "b"]
    assert len(b) == 4 and (b["value_vorp"] == 0).all()
    # Undrafted u: season 1 is his first NBA season.
    u = grid[grid.bbref_id == "u"]
    assert u["season"].tolist() == [2021, 2022] and u["season_num"].tolist() == [1, 2]


def test_outcomes_through_n_are_cumulative_and_same_point() -> None:
    otn = outcomes_through_n(season_grid(_players(), _seasons(), last_season=2022))
    a = otn[otn.bbref_id == "a"].set_index("n")
    assert a["total_vorp"].tolist() == [2.0, 5.0, 5.0, 6.0]
    assert a["seasons_in_nba"].tolist() == [1, 2, 2, 3]
    out = add_composites(otn, training_ids={"a", "b"})
    # At each N the training reference (a, b) has z-scores of +/-0.707 per metric.
    at2 = out[out.n == 2].set_index("bbref_id")["composite_vorp"]
    assert at2["a"] > 0 > at2["b"]
    assert at2[["a", "b"]].sum() == pytest.approx(0.0)


def test_tier_cut_maximizes_balanced_accuracy() -> None:
    from draft_dna.outcomes.tiers import best_cut

    lower = np.array([0.0, 0.1, 0.2, 0.9])
    upper = np.array([0.5, 0.8, 1.0, 1.2])
    assert best_cut(lower, upper) == 0.5


def test_tiers_assigned_by_cutoffs_and_increasing() -> None:
    from draft_dna.outcomes.tiers import TIERS, assign, calibrate

    peak = pd.Series([0.0, 0.0, 0.2, 0.3, 0.6, 0.7, 1.3, 1.5, 2.0, 2.2, 3.0, 3.5])
    anchor = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    cuts = calibrate(peak, anchor)
    assert cuts == sorted(cuts) and len(cuts) == len(TIERS) - 1
    assert assign(peak, cuts).tolist() == anchor.tolist()
