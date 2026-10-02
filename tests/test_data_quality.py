"""Data-quality tests against the built DuckDB database. Run with `make dq`.

Thresholds are deliberately explicit: if a source changes and coverage drops, a
test fails loudly instead of the model silently training on less data.
"""

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from draft_dna.config import get_settings
from tests.conftest import FIXTURES

pytestmark = pytest.mark.dq


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    path = get_settings().paths.database
    if not path.exists():
        pytest.skip("no database; run `make data` first")
    c = duckdb.connect(str(path), read_only=True)
    yield c
    c.close()


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    row = con.execute(sql).fetchone()
    assert row is not None
    return float(row[0])


# ----------------------------------------------------------------------- draft picks
def test_every_draft_class_present_with_plausible_size(con) -> None:
    counts = q(con, "SELECT draft_year, count(*) n FROM staging.bbref__draft_picks GROUP BY 1")
    years = set(get_settings().draft_classes.all_years())
    assert set(counts.draft_year) == years
    # 57-60 picks (forfeited picks and 29-team leagues reduce the count).
    assert counts.n.between(57, 60).all(), counts[~counts.n.between(57, 60)]


def test_draft_slots_unique(con) -> None:
    assert (
        scalar(
            con,
            """SELECT count(*) FROM (SELECT draft_year, pick_overall
        FROM staging.bbref__draft_picks GROUP BY 1, 2 HAVING count(*) > 1)""",
        )
        == 0
    )


def test_draft_sources_agree_on_names(con) -> None:
    # Nearly every pick links to stats.nba.com by name; low-similarity links are rare
    # and are only the pick-number fallback for garbled names ("Ha Ha", "Sun Sun").
    low = q(
        con,
        """SELECT draft_year, pick_overall, player_name, nba_player_name, nba_method, nba_score
        FROM modeled.xwalk__players WHERE nba_method LIKE 'draft%' AND nba_score < 75""",
    )
    total = scalar(con, "SELECT count(*) FROM modeled.xwalk__players WHERE drafted")
    assert len(low) / total < 0.005, low
    assert set(low.nba_method) <= {"draft_pick_slot"}, low


# ------------------------------------------------------------------------- crosswalk
def test_crosswalk_keys_unique(con) -> None:
    for col in ("bbref_id", "nba_person_id", "cbb_id", "bart_pid"):
        dupes = scalar(
            con,
            f"""SELECT count(*) FROM (SELECT {col} FROM modeled.xwalk__players
            WHERE {col} IS NOT NULL GROUP BY 1 HAVING count(*) > 1)""",
        )
        assert dupes == 0, f"{col} maps to more than one player"


def test_every_nba_player_has_nba_id(con) -> None:
    missing = q(
        con,
        """SELECT p.bbref_id, p.player_name, p.nba_method
        FROM modeled.xwalk__players p
        WHERE p.nba_person_id IS NULL AND EXISTS (
            SELECT 1 FROM modeled.core__nba_player_seasons s WHERE s.bbref_id = p.bbref_id)""",
    )
    assert missing.empty, missing


def test_college_prospects_linked_to_college_stats(con) -> None:
    rate = scalar(
        con,
        """SELECT avg((cbb_id IS NOT NULL)::INT) FROM modeled.xwalk__players
        WHERE drafted AND prospect_source = 'college'""",
    )
    assert rate >= 0.95, rate


def test_barttorvik_coverage_for_modern_college_picks(con) -> None:
    rate = scalar(
        con,
        """SELECT avg((bart_pid IS NOT NULL)::INT) FROM modeled.xwalk__players
        WHERE drafted AND prospect_source = 'college' AND draft_year >= 2010""",
    )
    assert rate >= 0.90, rate


def test_gold_set_crosswalk(con) -> None:
    gold = pd.read_csv(FIXTURES / "gold_crosswalk.csv", dtype=str, comment="#")
    xw = q(con, "SELECT bbref_id, nba_person_id, cbb_id, bart_pid FROM modeled.xwalk__players")
    xw = xw.astype({"nba_person_id": "Int64", "bart_pid": "Int64"}).astype(str).replace("<NA>", "")
    merged = gold.merge(xw, on="bbref_id", how="left", suffixes=("_gold", ""))
    errors = []
    for col in ("nba_person_id", "cbb_id", "bart_pid"):
        expected = merged[f"{col}_gold"].fillna("")
        actual = merged[col].fillna("")
        checked = expected != "?"  # '?' = not asserted for this player
        bad = merged[checked & (expected != actual)]
        errors += [
            f"{r.bbref_id} {col}: expected {r[col + '_gold']!r} got {r[col]!r}"
            for _, r in bad.iterrows()
        ]
    assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------- NBA seasons
def test_nba_seasons_unique_and_sane(con) -> None:
    assert (
        scalar(
            con,
            """SELECT count(*) FROM (SELECT bbref_id, season
        FROM modeled.core__nba_player_seasons GROUP BY 1, 2 HAVING count(*) > 1)""",
        )
        == 0
    )
    bad = q(
        con,
        """SELECT bbref_id, season, games, season_games FROM modeled.core__nba_player_seasons
        WHERE games > season_games + 4 OR mp < 0 OR games IS NULL""",
    )
    assert bad.empty, bad


def test_no_nba_seasons_before_draft(con) -> None:
    bad = q(
        con,
        """SELECT s.bbref_id, s.season, p.draft_year
        FROM modeled.core__nba_player_seasons s JOIN modeled.xwalk__players p USING (bbref_id)
        WHERE p.drafted AND s.season <= p.draft_year""",
    )
    assert bad.empty, bad


def test_award_counts_plausible(con) -> None:
    per_season = q(
        con,
        """SELECT season, sum((all_nba_team IS NOT NULL)::INT) all_nba,
        sum(all_star::INT) all_star FROM modeled.core__nba_player_seasons
        WHERE season < (SELECT max(season) FROM modeled.core__nba_player_seasons) GROUP BY 1""",
    )
    # Our universe excludes players drafted before 1996, so early seasons have few.
    assert (per_season.all_nba <= 15).all()
    assert (per_season.loc[per_season.season >= 2012, "all_nba"] >= 10).all(), per_season


# ------------------------------------------------------------------------- nulls
@pytest.mark.parametrize(
    ("table", "cols"),
    [
        ("modeled.xwalk__players", ["bbref_id", "player_name", "drafted"]),
        ("modeled.core__nba_player_seasons", ["bbref_id", "season", "games", "mp", "season_num"]),
        ("staging.bbref__draft_picks", ["draft_year", "round", "pick_overall", "bbref_id"]),
    ],
)
def test_key_columns_not_null(con, table: str, cols: list[str]) -> None:
    for c in cols:
        assert scalar(con, f"SELECT count(*) FROM {table} WHERE {c} IS NULL") == 0, f"{table}.{c}"


# --------------------------------------------------------------------- college
def test_derived_college_rates_match_published_where_both_exist(con) -> None:
    # Derived usage/assist rate fill pre-2010 gaps; they must reproduce the published
    # values wherever Sports-Reference has both.
    for stat in ("usg", "ast"):
        mae = scalar(
            con,
            f"""SELECT avg(abs({stat}_pct - {stat}_pct_derived))
            FROM modeled.core__college_player_seasons
            WHERE {stat}_pct_source = 'published' AND {stat}_pct_derived IS NOT NULL""",
        )
        assert mae < 0.5, f"{stat}: mean abs diff {mae:.2f}"


def test_barttorvik_attached_to_most_linked_college_seasons(con) -> None:
    rate = scalar(
        con,
        """SELECT avg((bart_bpm IS NOT NULL)::INT) FROM modeled.core__college_player_seasons
        WHERE season >= 2008 AND bart_pid IS NOT NULL""",
    )
    assert rate >= 0.90, rate


def test_post_draft_college_seasons_are_flagged(con) -> None:
    # Drafted-but-unsigned players can return to college (e.g. James Nnaji, drafted 2023,
    # Baylor 2025-26). Such seasons exist but must be flagged so features exclude them.
    bad = q(
        con,
        """SELECT c.bbref_id, c.season, p.draft_year
        FROM modeled.core__college_player_seasons c JOIN modeled.core__players p USING (bbref_id)
        WHERE p.drafted AND c.season > p.draft_year AND c.is_pre_draft""",
    )
    assert bad.empty, bad


# --------------------------------------------------------------------- outcomes
def test_outcomes_one_row_per_player_and_n(con) -> None:
    assert (
        scalar(
            con,
            """SELECT count(*) FROM (SELECT bbref_id, n FROM modeled.outcomes__outcomes_through_n
            GROUP BY 1, 2 HAVING count(*) > 1)""",
        )
        == 0
    )


def test_cumulative_outcomes_never_decrease(con) -> None:
    bad = q(
        con,
        """SELECT bbref_id, n FROM (
            SELECT bbref_id, n, minutes - lag(minutes) OVER w AS d_min,
                   seasons_in_nba - lag(seasons_in_nba) OVER w AS d_seasons,
                   peak3_blend - lag(peak3_blend) OVER w AS d_peak
            FROM modeled.outcomes__outcomes_through_n
            WINDOW w AS (PARTITION BY bbref_id ORDER BY n))
        WHERE d_min < 0 OR d_seasons < 0 OR d_peak < -1e-9""",
    )
    assert bad.empty, bad


def test_composites_centered_on_training_classes_at_every_n(con) -> None:
    # Same-point scale: at each N, training-class players average zero.
    worst = scalar(
        con,
        """SELECT max(abs(m)) FROM (SELECT o.n, avg(o.composite_blend) m
        FROM modeled.outcomes__outcomes_through_n o
        JOIN modeled.core__players p USING (bbref_id)
        WHERE p.draft_class_role = 'training' GROUP BY 1)""",
    )
    assert worst < 1e-6


def test_tiers_ordered_by_peak_value(con) -> None:
    t = q(
        con,
        """SELECT tier_idx, min(peak3_blend) lo, max(peak3_blend) hi
        FROM modeled.outcomes__outcomes_through_n GROUP BY 1 ORDER BY 1""",
    )
    assert t["tier_idx"].tolist() == list(range(6))
    assert (t["lo"].iloc[1:].to_numpy() >= t["hi"].iloc[:-1].to_numpy()).all()


def test_censoring_rules(con) -> None:
    c = q(con, "SELECT * FROM modeled.outcomes__careers")
    never = c[c["seasons_played"] == 0]
    assert (never["ended"] & (never["duration"] == 0)).all()
    last_complete = get_settings().current_nba_season - 1
    active = c[c["last_season"] >= last_complete - 1]
    assert not active["ended"].any()


# ------------------------------------------------------------------ projections
def test_projections_are_as_of_draft_night(con) -> None:
    bad = q(
        con,
        """SELECT player_name, draft_year, max_train_class FROM modeled.projections__projections
        WHERE max_train_class + 6 > draft_year""",
    )
    assert bad.empty, bad


def test_projection_bands_ordered_and_tiers_sum_to_one(con) -> None:
    p = q(con, "SELECT * FROM modeled.projections__projections")
    assert (p["floor"] <= p["median"]).all() and (p["median"] <= p["ceiling"]).all()
    tiers = p[["p_out_of_league", "p_bust", "p_rotation", "p_starter", "p_all_star", "p_all_nba"]]
    assert np.allclose(tiers.sum(axis=1), 1.0)
    assert p["bbref_id"].is_unique


def test_comps_come_from_earlier_classes_with_known_outcomes(con) -> None:
    bad = q(
        con,
        """SELECT c.bbref_id, c.comp_id FROM modeled.projections__comps c
        JOIN modeled.core__players p USING (bbref_id)
        WHERE c.comp_draft_year >= p.draft_year OR c.comp_peak6 IS NULL
           OR c.comp_id = c.bbref_id""",
    )
    assert bad.empty, bad
    per = q(con, "SELECT bbref_id, count(*) n FROM modeled.projections__comps GROUP BY 1")
    assert (per["n"] == 15).all()


# -------------------------------------------------------------------- shot DNA
def test_espn_links_unique_and_pre_draft_shots_only(con) -> None:
    assert (
        scalar(
            con,
            """SELECT count(*) FROM (SELECT season, athlete_id FROM modeled.shots__espn_player_map
            GROUP BY 1, 2 HAVING count(*) > 1)""",
        )
        == 0
    )
    bad = q(
        con,
        """SELECT s.bbref_id, s.season, p.draft_year FROM modeled.shots__college_shots s
        JOIN modeled.core__players p USING (bbref_id) WHERE s.season > p.draft_year""",
    )
    assert bad.empty, bad


def test_shot_mix_and_zone_shares_sum_to_one(con) -> None:
    mix = q(con, "SELECT rim_rate + mid_rate + three_rate AS t FROM modeled.features__shot_dna")
    assert np.allclose(mix["t"].dropna(), 1.0)
    zones = q(con, "SELECT * FROM modeled.features__shot_zones")
    assert np.allclose(zones.filter(like="zone_").sum(axis=1), 1.0)


def test_eb_shrunk_rates_are_valid_probabilities(con) -> None:
    eb = q(con, "SELECT rim_fg_eb, mid_fg_eb, three_fg_eb FROM modeled.features__shot_dna")
    assert ((eb.dropna() > 0) & (eb.dropna() < 1)).all().all()


def test_spatial_eligibility_rule(con) -> None:
    bad = q(
        con,
        """SELECT * FROM modeled.features__shot_coverage WHERE xy_eligible
        AND (xy_fga < 100 OR xy_coverage < 0.4)""",
    )
    assert bad.empty, bad
    maps = scalar(con, "SELECT count(*) FROM modeled.features__shot_maps")
    elig = scalar(con, "SELECT count(*) FROM modeled.features__shot_coverage WHERE xy_eligible")
    assert maps == elig


# ----------------------------------------------------------------------- phase 6
def test_pit_in_unit_interval_and_verdicts_consistent(con) -> None:
    p = q(con, "SELECT * FROM modeled.phase6__player_outcomes_vs_projection")
    for n in (4, 6, 8):
        d = p[p[f"pit{n}"].notna()]
        assert d[f"pit{n}"].between(0, 1).all()
        assert (
            d.loc[d[f"verdict{n}"] == "beat ceiling", f"actual{n}"]
            > d.loc[d[f"verdict{n}"] == "beat ceiling", f"ceiling{n}"]
        ).all()
        assert (
            d.loc[d[f"verdict{n}"] == "below floor", f"actual{n}"]
            < d.loc[d[f"verdict{n}"] == "below floor", f"floor{n}"]
        ).all()


def test_projections_used_for_verdicts_are_well_calibrated_overall(con) -> None:
    # If draft-night ranges are calibrated, mean PIT ~ 0.5 (Phase 6 analyses rely on this).
    p = q(con, "SELECT pit4, pit6, pit8 FROM modeled.phase6__player_outcomes_vs_projection")
    for col in p.columns:
        assert abs(p[col].mean() - 0.5) < 0.03, (col, p[col].mean())


# ----------------------------------------------------------------------- phase 7
def test_grades_ordered_and_consistent_with_status(con) -> None:
    g = q(con, "SELECT * FROM modeled.grading__grades")
    assert g["bbref_id"].is_unique
    for kind in ("projected", "current"):
        assert (g[f"{kind}_floor"] <= g[f"{kind}_median"] + 1e-9).all()
        assert (g[f"{kind}_median"] <= g[f"{kind}_ceiling"] + 1e-9).all()
    assert g["confidence"].between(0, 1).all()
    assert (g.loc[g["status"] == "Projection", "grade"] == "-").all()
    assert g.loc[g["status"] != "Projection", "grade"].isin(list("ABCD")).all()


def test_grade_ranges_respect_peak_already_reached(con) -> None:
    # A best-3-season value can't fall: the current floor is at least what he has reached.
    d = q(
        con,
        """SELECT g.current_floor, o.peak3_blend
           FROM modeled.grading__grades g
           JOIN modeled.outcomes__outcomes_through_n o
             ON o.bbref_id = g.bbref_id AND o.n = g.seasons
           WHERE g.seasons > 0""",
    )
    assert len(d) > 500
    assert (d["current_floor"] >= d["peak3_blend"] - 1e-6).all()


def test_plays_like_excludes_self_and_respects_height(con) -> None:
    d = q(
        con,
        """SELECT p.bbref_id, p.plays_like_id, a.height_in AS h1, b.height_in AS h2
           FROM modeled.grading__plays_like p
           LEFT JOIN modeled.features__predraft a ON a.bbref_id = p.bbref_id
           LEFT JOIN modeled.features__predraft b ON b.bbref_id = p.plays_like_id""",
    )
    assert (d["bbref_id"] != d["plays_like_id"]).all()
    both = d.dropna(subset=["h1", "h2"])
    assert ((both["h1"] - both["h2"]).abs() <= 3.0).all()


def test_rookie_tracker_covers_live_class() -> None:
    from draft_dna.grading import tracker

    s = get_settings()
    t = tracker.rookie_tracker(s)
    assert len(t) >= 58  # two rounds, minus forfeited picks
    assert (t["projected_floor"] <= t["projected_ceiling"]).all()
    assert (
        t["status"]
        .isin(["no games yet", "pacing above ceiling", "pacing below floor", "within range"])
        .all()
    )
