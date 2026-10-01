from draft_dna.ingest.bbref import parse_draft, parse_player_page
from tests.conftest import FIXTURES

BB = FIXTURES / "bbref"


def test_parse_draft_assigns_rounds_from_headers() -> None:
    df = parse_draft((BB / "draft_2019.html").read_bytes(), 2019)
    assert list(df["pick_overall"]) == [1, 2, 30, 31, 32, 60]
    assert list(df["round"]) == [1, 1, 1, 2, 2, 2]
    first = df.iloc[0]
    assert (first.player_name, first.bbref_id, first.team_id, first.college_name) == (
        "Zion Williamson",
        "willizi01",
        "NOP",
        "Duke",
    )
    assert (df["draft_year"] == 2019).all()


def test_parse_player_page_bio() -> None:
    bio = parse_player_page((BB / "player_willizi01.html").read_bytes(), "willizi01")
    assert bio["birth_date"] == "2000-07-06"
    assert bio["height_in"] == 78
    assert bio["weight_lb"] == 284
    assert bio["colleges"] == "Duke"
    assert bio["recruit_rank"] == 4
    assert bio["birthplace"] == "Salisbury, North Carolina"
    assert bio["cbb_id"] == "zion-williamson-1"
    assert "1st overall" in bio["draft_text"]
