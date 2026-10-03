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


def test_rights_team_follows_trade_chain_from_current_holder() -> None:
    import pandas as pd

    from draft_dna.modeled.integrate import rights_team

    trades = pd.DataFrame({"team_from": ["LAL", "NYK", "OKC"], "team_to": ["OKC", "BOS", "MIN"]})
    # LAL -> OKC -> MIN; the NYK -> BOS leg belongs to someone else in a multi-team deal.
    assert rights_team("LAL", trades) == "MIN"
    assert rights_team("CLE", trades.iloc[0:0]) == "CLE"


def test_parse_transactions_reads_team_codes_and_multi_team_trades() -> None:
    from draft_dna.ingest.bbref import parse_transactions

    html = b"""<div id="div_transactions">
    <p class="transaction "><strong>June 21, 2018</strong>: Drafted by the
      <a data-attr-to="CHO" href="#">Charlotte Hornets</a>.</p>
    <p class="transaction "><strong>June 21, 2018</strong>: Traded by the
      <a data-attr-from="CHO" href="#">Hornets</a> to the
      <a data-attr-to="LAC" href="#">Clippers</a>.</p>
    <p class="transaction "><strong>November 20, 2020</strong>: As part of a 3-team trade, traded by
      the <a data-attr-from="OKC" href="#">Thunder</a> to the
      <a data-attr-to="MIN" href="#">Wolves</a>.</p>
    </div>"""
    t = parse_transactions(html, "x")
    assert t["kind"].tolist() == ["drafted", "traded", "traded"]
    assert t["team_to"].tolist() == ["CHO", "LAC", "MIN"]
    assert t.loc[1, "team_from"] == "CHO"


def test_class_metrics_curve_is_monotone_and_average_class_scores_zero() -> None:
    import numpy as np
    import pandas as pd

    from draft_dna.grading import classes as C

    rng = np.random.default_rng(0)
    rows = []
    for year in range(1996, 2018):
        for pick in range(1, 61):
            v = max(0.0, 3.0 / pick**0.6 + rng.normal(0, 0.3))
            rows.append(
                {
                    "bbref_id": f"{year}-{pick}",
                    "player_name": f"p{pick}",
                    "draft_year": year,
                    "pick": pick,
                    "current_median": v,
                    "status": "Career Grade",
                    "current_tier": "Starter",
                    **{f"post_q{q:.2f}": v for q in np.arange(0.05, 0.951, 0.05)},
                }
            )
    g = pd.DataFrame(rows)
    curve = C.typical_curve(g)
    assert (np.diff(curve["typical_peak"]) <= 1e-9).all()
    strength = C.class_strength(g, curve)
    assert abs(strength["strength"].mean()) < 1.0  # average class ~ 0
    eq = C.equivalent_pick(pd.Series([10.0, curve["typical_peak"].iloc[9], 0.0]), curve)
    assert eq.iloc[0] == 1 and eq.iloc[1] <= 10 and np.isnan(eq.iloc[2])


def test_player_metrics_stash_filter_shrinkage_and_pick_value() -> None:
    import numpy as np
    import pandas as pd

    from draft_dna.grading import classes as C
    from draft_dna.grading import player_metrics as PM

    # Late bloomer: a stash player (0 early NBA seasons) is excluded.
    peaks = pd.DataFrame({3: [0.5, 1.0, 1.5, 0.0], 8: [0.6, 1.2, 1.6, 2.0]}, index=list("abcs"))
    early = pd.Series({"a": 3, "b": 3, "c": 2, "s": 0})
    lb = PM.late_bloomer(peaks, early)
    assert "s" not in lb.index and len(lb) == 3

    # Playoff riser: same gap, more playoff minutes -> larger (less shrunk) score.
    d = pd.DataFrame(
        {
            "bbref_id": ["x", "y"],
            "po_mp": [400.0, 2000.0],
            "po_bpm": [3.0, 3.0],
            "bpm": [1.0, 1.0],
        }
    )
    r = PM.playoff_riser(d)
    assert 0 < r["x"] < r["y"] < 2.0

    curve = pd.DataFrame(
        {"pick": np.arange(1, 61), "average_peak": np.linspace(3, 0.1, 60), "typical_peak": 0}
    )
    pv = C.pick_value(curve)
    assert pv["value"].iloc[0] == 100 and (np.diff(pv["value"]) <= 0).all()


def test_honor_tiers_ladder() -> None:
    from draft_dna.outcomes.tiers import honor_tier

    assert honor_tier(all_nba=21, first_team=13, mvps=4) == "Legend"  # LeBron
    assert honor_tier(all_nba=11, first_team=4, mvps=0) == "Legend"  # Chris Paul (10+ All-NBA)
    assert honor_tier(all_nba=1, first_team=1, mvps=1) == "MVP"  # Derrick Rose
    assert honor_tier(all_nba=6, first_team=6, mvps=0) == "Superstar"  # Luka
    assert honor_tier(all_nba=5, first_team=1, mvps=0) is None  # stays All-NBA


def test_parse_trades_splits_sides_and_resolves_future_picks() -> None:
    from draft_dna.grading.pick_trades import parse_trades

    html = b"""<div id="div_transactions">
    <p class="transaction "><strong>June 21, 2018</strong>: Traded by the
      <a data-attr-from="ATL" href="/teams/ATL/2018.html">Hawks</a> with
      <a href="/players/x/extraxx01.html">Extra Guy</a> to the
      <a data-attr-to="DAL" href="/teams/DAL/2018.html">Mavericks</a> for
      <a href="/players/y/youngtr01.html">Trae Young</a> and a 2019 1st round draft pick
      (<a href="/players/r/reddica01.html">Cam Reddish</a> was later selected) and a 2030
      2nd round draft pick.</p>
    <p class="transaction "><strong>June 22, 2018</strong>: As part of a 3-team trade,
      traded by the <a data-attr-from="X" href="#">X</a> to the
      <a data-attr-to="Y" href="#">Y</a>.</p>
    </div>"""
    t = parse_trades(html, "doncilu01")
    assert t[0]["sent"] == ["doncilu01", "extraxx01"]
    assert t[0]["received"] == ["reddica01", "youngtr01"]
    assert t[0]["received_unresolved"] == 1  # the 2030 pick hasn't become a player
    assert t[0]["team_from"] == "ATL" and t[0]["team_to"] == "DAL"
    assert t[1]["multi_team"]
