import pandas as pd

from draft_dna.ingest.espn import map_schools

TEAMS = pd.DataFrame(
    {
        "espn_team_id": [193, 2390, 2350, 41],
        "location": ["Miami (OH)", "Miami", "Loyola Chicago", "UConn"],
    }
)


def test_parenthetical_qualifiers_do_not_collapse_schools() -> None:
    m = map_schools(
        ["miami-fl", "miami-oh"], {"miami-fl": "Miami (FL)", "miami-oh": "Miami (OH)"}, TEAMS
    ).set_index("school_slug")["espn_team_id"]
    assert m["miami-fl"] == 2390 and m["miami-oh"] == 193


def test_qualifier_rule_without_override() -> None:
    # A school whose name lacks "(OH)" must never map to "Miami (OH)".
    m = map_schools(["miami-fake"], {"miami-fake": "Miami"}, TEAMS)
    assert m.iloc[0]["espn_team_id"] == 2390
