"""Data-quality tests against the built DuckDB database. Run with `make dq`.

Thresholds are deliberately explicit: if a source changes and coverage drops, a
test fails loudly instead of the model silently training on less data.
"""

from collections.abc import Iterator

import duckdb
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
        errors += [f"{r.bbref_id} {col}: expected {r[col + '_gold']!r} got {r[col]!r}"
                   for _, r in bad.iterrows()]  # fmt: skip
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
