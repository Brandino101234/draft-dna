import pandas as pd

from draft_dna.ingest.barttorvik import parse_player_csv, parse_team_csv
from tests.conftest import FIXTURES


def test_player_columns_line_up() -> None:
    df = parse_player_csv((FIXTURES / "barttorvik" / "players_sample.csv").read_bytes())
    zion = df[df.player_name == "Zion Williamson"].iloc[0]
    assert (zion.team, zion.year, zion.nba_pick, zion.birth_date) == (
        "Duke",
        "2019",
        "1",
        "2000-07-06",
    )
    # Shot splits must partition two-point attempts.
    assert int(zion.rim_att) + int(zion.mid_att) == int(zion.two_pa)
    assert int(zion.rim_made) + int(zion.mid_made) == int(zion.two_pm)
    beasley = df[df.player_name == "Michael Beasley"].iloc[0]
    assert (beasley.nba_pick, beasley.birth_date, beasley.rim_att) == ("2", "1988-10-15", "")


def test_team_csv_header_quirk() -> None:
    body = b'rank,team,"Fun Rk, adjt"\n1,Gonzaga,30,70.1\n'
    df = parse_team_csv(body)
    assert list(df.columns) == ["rank", "team", "fun_rk", "adjt"]
    assert df.iloc[0].adjt == "70.1"
    assert isinstance(df, pd.DataFrame)
