import pandas as pd
import pytest

from draft_dna.crosswalk.build import link_nba, match_bart, normalize_school, school_sim
from draft_dna.crosswalk.names import normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Luka Dončić", "luka doncic"),
        ("Kevin Porter Jr.", "kevin porter"),
        ("D'Angelo Russell", "dangelo russell"),
        ("P.J. Tucker", "pj tucker"),
        ("Glenn Robinson III", "glenn robinson"),
        ("Nenê Hilário", "nene"),
        ("Ron Artest", "metta world peace"),
        ("Shai Gilgeous-Alexander", "shai gilgeous alexander"),
        (None, ""),
    ],
)
def test_normalize_name(raw: str | None, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_suffix_kept_when_requested() -> None:
    assert normalize_name("Gary Payton II", drop_suffix=False) == "gary payton ii"


@pytest.mark.parametrize(
    ("a", "b"),
    [("Kansas St.", "Kansas State"), ("UConn", "Connecticut"), ("Saint Mary's", "St. Mary's (CA)")],
)
def test_school_names_match_across_sources(a: str, b: str) -> None:
    assert school_sim(a, b) >= 85, (normalize_school(a), normalize_school(b))


def test_school_names_differ() -> None:
    assert school_sim("Kansas", "Kansas State") < 100


def _universe(**overrides: object) -> pd.DataFrame:
    base = {
        "bbref_id": "x01",
        "player_name": "Mike James",
        "nba_person_id": pd.NA,
        "nba_player_name": None,
        "first_season": 2018,
    }
    base.update(overrides)
    return pd.DataFrame([base])


NBA = pd.DataFrame(
    {
        "nba_person_id": [1, 2, 3],
        "player_name": ["Mike James", "Mike James", "Mikal Bridges"],
        "first_season": [2002, 2018, 2019],
    }
)


def test_undrafted_link_disambiguates_shared_names_by_debut_season() -> None:
    out = link_nba(_universe(), NBA).iloc[0]
    assert (out.nba_person_id, out.nba_method) == (2, "name_season")


def test_drafted_link_uses_draft_slot_and_scores_name() -> None:
    u = _universe(nba_person_id=3, nba_player_name="Mikal Bridges", player_name="Mikal Bridges")
    out = link_nba(u, NBA).iloc[0]
    assert (out.nba_person_id, out.nba_method, out.nba_score) == (3, "draft_slot", 100.0)


def test_unmatched_name_is_unresolved() -> None:
    out = link_nba(_universe(player_name="Nobody Here"), NBA).iloc[0]
    assert out.nba_method == "unresolved"


BART = pd.DataFrame(
    {
        "bart_pid": [10, 11, 12],
        "player_name": ["Marcus Williams", "Marcus Williams", "Marcus Thornton"],
        "season": [2007, 2008, 2009],
        "team": ["Connecticut", "Arizona", "LSU"],
        "birth_date": [pd.NaT, pd.Timestamp("1986-12-03"), pd.NaT],
        "nba_pick": pd.array([pd.NA, 33, pd.NA], dtype="Int64"),
    }
)
BART["last_norm"] = BART["player_name"].map(lambda n: normalize_name(n).split()[-1])
BART["first_norm"] = BART["player_name"].map(lambda n: normalize_name(n).split()[0])


def _prospect(**kw: object) -> pd.Series:
    base = {
        "player_name": "Marcus Williams",
        "player_first": "marcus",
        "player_last": "williams",
        "drafted": True,
        "draft_year": 2008,
        "pick_overall": 33,
        "first_season": 2009,
        "birth_date": pd.NaT,
        "colleges": "Arizona",
    }
    base.update(kw)
    return pd.Series(base)


def test_bart_prefers_birthdate() -> None:
    m = match_bart(_prospect(birth_date=pd.Timestamp("1986-12-03")), BART)
    assert (m.bart_pid, m.method) == (11, "birthdate")


def test_bart_uses_nba_pick_when_no_birthdate() -> None:
    m = match_bart(_prospect(), BART)
    assert (m.bart_pid, m.method) == (11, "nba_pick")


def test_bart_same_name_different_school_not_confused() -> None:
    m = match_bart(_prospect(pick_overall=99, colleges="Connecticut"), BART)
    assert (m.bart_pid, m.method) == (10, "name_school")
