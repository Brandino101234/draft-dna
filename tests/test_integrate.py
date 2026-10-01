import pandas as pd

from draft_dna.modeled.integrate import BART_COLS, attach_bart_seasons


def _bart(rows: list[tuple[int, str, int, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["bart_pid", "player_name", "season", "team", "bpm"])
    for c in BART_COLS:
        if c not in df:
            df[c] = 0.0
    return df


def test_transfer_seasons_get_barttorvik_stats_from_each_school() -> None:
    universe = pd.DataFrame({"bbref_id": ["martica02"], "player_name": ["Caleb Martin"]})
    seasons = pd.DataFrame(
        {
            "bbref_id": ["martica02"] * 3,
            "bart_pid": pd.array([31] * 3, dtype="Int64"),  # crosswalk keeps the latest ID
            "season": [2016, 2018, 2019],
            "team_name_abbr": ["NC State", "Nevada", "Nevada"],
        }
    )
    bart = _bart(
        [
            (30, "Caleb Martin", 2016, "N.C. State", 1.0),
            (31, "Caleb Martin", 2018, "Nevada", 5.0),
            (31, "Caleb Martin", 2019, "Nevada", 6.0),
            (99, "Caleb Martin", 2016, "Some Other U", -3.0),  # namesake elsewhere
        ]
    )
    out = attach_bart_seasons(seasons, universe, bart)
    assert out["bart_bpm"].tolist() == [1.0, 5.0, 6.0]
    assert out["bart_season_pid"].tolist() == [30, 31, 31]


def test_wrong_school_is_not_attached() -> None:
    universe = pd.DataFrame({"bbref_id": ["x"], "player_name": ["Chris Wright"]})
    seasons = pd.DataFrame(
        {
            "bbref_id": ["x"],
            "bart_pid": pd.array([pd.NA], dtype="Int64"),
            "season": [2011],
            "team_name_abbr": ["Dayton"],
        }
    )
    bart = _bart([(5, "Chris Wright", 2011, "Georgetown", 2.0)])
    out = attach_bart_seasons(seasons, universe, bart)
    assert out["bart_bpm"].isna().all()


def test_derived_usage_matches_formula_and_keeps_published_values() -> None:
    from draft_dna.modeled.integrate import derive_rates

    df = pd.DataFrame(
        {
            "fga": [400.0, 400.0], "fta": [100.0, 100.0], "tov": [60.0, 60.0],
            "mp": [1000.0, 1000.0], "ast": [100.0, 100.0], "fg": [180.0, 180.0],
            "tm_g": [30, 30], "tm_mp": [6000.0, None], "tm_fga": [1800.0, 1800.0],
            "tm_fta": [600.0, 600.0], "tm_tov": [400.0, 400.0], "tm_fg": [800.0, 800.0],
            "usg_pct": [25.0, None], "ast_pct": [None, None],
        }
    )  # fmt: skip
    out = derive_rates(df)
    # Row 0 keeps the published usage; row 1 derives it with team minutes = 30 * 200.
    assert out.loc[0, "usg_pct"] == 25.0 and out.loc[0, "usg_pct_source"] == "published"
    expected = 100 * (400 + 44 + 60) * 1200 / (1000 * (1800 + 264 + 400))
    assert abs(out.loc[1, "usg_pct"] - expected) < 1e-9
    assert out.loc[1, "usg_pct_source"] == "derived" and bool(out.loc[1, "tm_mp_estimated"])
    assert abs(out.loc[1, "ast_pct"] - 100 * 100 / ((1000 / 1200) * 800 - 180)) < 1e-9
