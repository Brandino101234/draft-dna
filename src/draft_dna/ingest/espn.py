"""ESPN men's college basketball play-by-play (unofficial public JSON API).

Used for college shot data: every field-goal attempt carries a shot type (layup, dunk,
jumper, three, tip) and, for some games only, x/y court coordinates. This is ESPN's
unofficial site API (the same one cbbpy uses); it is not a licensed feed, so access is
polite (1 request/second), cached, and for personal, non-commercial research only.

`season` is the ending year (2019 = 2018-19), matching the rest of the project.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from draft_dna.ingest.fetcher import Fetcher, NotFoundError
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

SOURCE = "espn"
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
# ESPN marks a missing coordinate with huge negative sentinels (e.g. -214748340).
COORD_LIMIT = 100


def _json(fetcher: Fetcher, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    body = fetcher.get(url, params, ext="json")
    result: dict[str, Any] = json.loads(body)
    return result


def teams(fetcher: Fetcher) -> pd.DataFrame:
    data = _json(fetcher, f"{BASE}/teams", {"limit": 1000})
    rows = []
    for t in data["sports"][0]["leagues"][0]["teams"]:
        team = t["team"]
        rows.append(
            {"espn_team_id": int(team["id"]), "location": team.get("location"),
             "display_name": team.get("displayName"), "abbrev": team.get("abbreviation")}
        )  # fmt: skip
    return pd.DataFrame(rows)


def schedule(fetcher: Fetcher, team_id: int, season: int) -> pd.DataFrame:
    try:
        data = _json(fetcher, f"{BASE}/teams/{team_id}/schedule", {"season": season})
    except NotFoundError:
        return pd.DataFrame()
    rows = []
    for e in data.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        broadcasts = [b.get("media", {}).get("shortName") for b in comp.get("broadcasts", []) or []]
        rows.append({"espn_team_id": team_id, "season": season, "game_id": int(e["id"]),
                     "date": e.get("date"), "name": e.get("name"),
                     "broadcast": ",".join(b for b in broadcasts if b) or None})  # fmt: skip
    return pd.DataFrame(rows)


def _valid(c: dict[str, Any] | None) -> bool:
    if not c or "x" not in c or "y" not in c:
        return False
    return bool(abs(c["x"]) < COORD_LIMIT and abs(c["y"]) < COORD_LIMIT)


def game_shots(fetcher: Fetcher, game_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(field-goal attempts, roster, game info) for one game."""
    try:
        g = _json(fetcher, f"{BASE}/summary", {"event": game_id})
    except NotFoundError:
        return pd.DataFrame(), pd.DataFrame(), {"game_id": game_id, "available": False}
    shots = []
    for p in g.get("plays", []) or []:
        if not p.get("shootingPlay"):
            continue
        ptype = (p.get("type") or {}).get("text", "")
        if "free" in ptype.lower() or "free throw" in (p.get("text") or "").lower():
            continue
        c = p.get("coordinate")
        athletes = [a.get("athlete", {}).get("id") for a in p.get("participants", []) or []]
        shots.append({
            "game_id": game_id,
            "team_id": int((p.get("team") or {}).get("id", -1)),
            "athlete_id": int(athletes[0]) if athletes and athletes[0] else None,
            "assist_athlete_id": int(athletes[1]) if len(athletes) > 1 and athletes[1] else None,
            "period": (p.get("period") or {}).get("number"),
            "clock": (p.get("clock") or {}).get("displayValue"),
            "shot_type": ptype,
            "text": p.get("text"),
            "made": bool(p.get("scoringPlay")),
            "score_value": p.get("scoreValue"),
            "x": c.get("x") if _valid(c) else None,
            "y": c.get("y") if _valid(c) else None,
        })  # fmt: skip
    roster = []
    for team in (g.get("boxscore") or {}).get("players", []) or []:
        tid = int(team.get("team", {}).get("id", -1))
        for stat in team.get("statistics", []) or []:
            for a in stat.get("athletes", []) or []:
                ath = a.get("athlete", {})
                if not ath.get("id"):  # occasional placeholder rows ("Team", unnamed)
                    continue
                roster.append({"game_id": game_id, "team_id": tid, "athlete_id": int(ath["id"]),
                               "athlete_name": ath.get("displayName")})  # fmt: skip
    header = (g.get("header") or {}).get("competitions", [{}])[0]
    comps = header.get("competitors", [])
    home: dict[str, Any] = next((c for c in comps if c.get("homeAway") == "home"), {})
    info = {
        "game_id": game_id,
        "available": True,
        "n_plays": len(g.get("plays", []) or []),
        "neutral_site": bool(header.get("neutralSite")),
        "conference_game": bool(header.get("conferenceCompetition")),
        "home_team_id": int(home.get("id", -1)) if home else -1,
    }
    return pd.DataFrame(shots), pd.DataFrame(roster).drop_duplicates(), info


# Sports-Reference school slug -> ESPN team id, where names don't match automatically.
SCHOOL_OVERRIDES = {
    "illinois-chicago": 82, "loyola-il": 2350, "appalachian-state": 2026, "loyola-md": 2352,
    "college-of-charleston": 232, "tennessee-martin": 2630, "southern-mississippi": 2572,
}  # fmt: skip


def map_schools(slugs: list[str], names: dict[str, str], espn_teams: pd.DataFrame) -> pd.DataFrame:
    """Best ESPN team for each Sports-Reference school (full name or slug words)."""
    from draft_dna.crosswalk.build import school_sim

    rows = []
    for slug in slugs:
        if slug in SCHOOL_OVERRIDES:
            rows.append((slug, SCHOOL_OVERRIDES[slug], 100.0, "manual"))
            continue
        cands = [x for x in (names.get(slug), slug.replace("-", " ")) if x]
        scores = espn_teams["location"].map(lambda loc, c=cands: max(school_sim(x, loc) for x in c))
        best = scores.idxmax()
        rows.append((slug, int(espn_teams.loc[best, "espn_team_id"]), float(scores[best]), "name"))
    out = pd.DataFrame(rows, columns=["school_slug", "espn_team_id", "score", "method"])
    return out[out["score"] >= 95]
