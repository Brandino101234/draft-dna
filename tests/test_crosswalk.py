import pandas as pd
import pytest

from draft_dna.crosswalk.build import (
    link_draft_history,
    link_undrafted,
    match_bart,
    normalize_school,
    school_sim,
)
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
    base = {"bbref_id": "x01", "player_name": "Mike James", "first_season": 2018,
            "birth_date": pd.NaT}  # fmt: skip
    base.update(overrides)
    return pd.DataFrame([base])


NBA = pd.DataFrame(
    {
        "nba_person_id": [1, 2, 3, 4, 5, 6],
        "player_name": [
            "Mike James",
            "Mike James",
            "Mikal Bridges",
            "Ike Fontaine",
            "Tony Mitchell",
            "Tony Mitchell",
        ],
        "first_season": [2002, 2018, 2019, 2011, 2014, 2014],
    }
)


def test_undrafted_link_disambiguates_shared_names_by_debut_season() -> None:
    out = link_undrafted(_universe(), NBA).iloc[0]
    assert (out.nba_person_id, out.nba_method) == (2, "name_season")


def test_same_name_same_season_split_by_birthdate() -> None:
    births = {5: "1989-04-07", 6: "1992-04-07"}
    u = _universe(
        player_name="Tony Mitchell", first_season=2014, birth_date=pd.Timestamp("1992-04-07")
    )
    out = link_undrafted(u, NBA, birth_lookup=births.get).iloc[0]
    assert (out.nba_person_id, out.nba_method) == (6, "name_birthdate")


def test_nickname_linked_by_unique_last_name_and_initial() -> None:
    out = link_undrafted(_universe(player_name="Isaac Fontaine", first_season=2011), NBA).iloc[0]
    assert (out.nba_person_id, out.nba_method) == (4, "lastname_season")


def test_last_name_alone_never_links_different_first_initials() -> None:
    nba = pd.DataFrame(
        {"nba_person_id": [7], "player_name": ["Davon Reed"], "first_season": [2018]}
    )
    out = link_undrafted(_universe(player_name="Willie Reed", first_season=2016), nba).iloc[0]
    assert out.nba_method == "unresolved"


def test_unmatched_name_is_unresolved() -> None:
    out = link_undrafted(_universe(player_name="Nobody Here"), NBA).iloc[0]
    assert out.nba_method == "unresolved"


def test_draft_link_survives_pick_numbering_differences() -> None:
    # Sources disagree on pick numbers (2001: BBRef #30 Hassell, stats.nba.com #30 Arenas).
    picks = pd.DataFrame(
        {
            "bbref_id": ["hasse01", "arena01", "sweet01"],
            "player_name": ["Trenton Hassell", "Gilbert Arenas", "Mike Sweetney"],
            "draft_year": [2001, 2001, 2001],
            "pick_overall": [30, 31, 9],
        }
    )
    hist = pd.DataFrame(
        {
            "draft_year": [2001, 2001, 2001],
            "pick_overall": [30, 29, 9],
            "nba_person_id": [10, 11, 12],
            "nba_player_name": ["Gilbert Arenas", "Trenton Hassell", "Michael Sweetney"],
            "pre_draft_org": ["Arizona", "Austin Peay", "Georgetown"],
            "pre_draft_org_type": ["College/University"] * 3,
        }
    )
    out = link_draft_history(picks, hist).set_index("bbref_id")
    assert out.loc["hasse01", "nba_person_id"] == 11
    assert out.loc["arena01", "nba_person_id"] == 10
    assert out.loc["sweet01", ["nba_person_id", "draft_link_method"]].tolist() == [12, "fuzzy_name"]


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


def test_bart_nickname_accepted_with_birthdate_and_last_name() -> None:
    bart = pd.DataFrame({"bart_pid": [20], "player_name": ["Edrice Adebayo"], "season": [2017],
                         "team": ["Kentucky"], "birth_date": [pd.Timestamp("1997-07-18")],
                         "nba_pick": pd.array([14], dtype="Int64")})  # fmt: skip
    bart["last_norm"] = "adebayo"
    bart["first_norm"] = "edrice"
    p = _prospect(player_name="Bam Adebayo", player_first="bam", player_last="adebayo",
                  draft_year=2017, pick_overall=14, birth_date=pd.Timestamp("1997-07-18"),
                  colleges="Kentucky")  # fmt: skip
    m = match_bart(p, bart)
    assert (m.bart_pid, m.method) == (20, "birthdate")


def test_bart_transfer_with_two_ids_resolves_to_latest_school() -> None:
    bart = pd.DataFrame(
        {
            "bart_pid": [30, 30, 31, 31],
            "player_name": ["Caleb Martin"] * 4,
            "season": [2015, 2016, 2018, 2019],
            "team": ["N.C. State", "N.C. State", "Nevada", "Nevada"],
            "birth_date": [pd.Timestamp("1995-09-28")] * 4,
            "nba_pick": pd.array([pd.NA] * 4, dtype="Int64"),
        }
    )
    bart["last_norm"] = "martin"
    bart["first_norm"] = "caleb"
    p = _prospect(player_name="Caleb Martin", player_first="caleb", player_last="martin",
                  drafted=False, first_season=2020, birth_date=pd.Timestamp("1995-09-28"),
                  colleges="NC State,Nevada")  # fmt: skip
    m = match_bart(p, bart)
    assert (m.bart_pid, m.method) == (31, "birthdate_transfer")


def test_twins_at_same_school_are_not_merged() -> None:
    bart = pd.DataFrame(
        {
            "bart_pid": [40, 41],
            "player_name": ["Travis Wear", "David Wear"],
            "season": [2014, 2014],
            "team": ["UCLA", "UCLA"],
            "birth_date": [pd.Timestamp("1990-09-21")] * 2,
            "nba_pick": pd.array([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    bart["last_norm"] = "wear"
    bart["first_norm"] = ["travis", "david"]
    p = _prospect(player_name="Travis Wear", player_first="travis", player_last="wear",
                  drafted=False, first_season=2015, birth_date=pd.Timestamp("1990-09-21"),
                  colleges="UNC,UCLA")  # fmt: skip
    assert match_bart(p, bart).bart_pid == 40
