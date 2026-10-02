"""Link ESPN athlete ids to bbref_id, one college season at a time.

For each drafted player's pre-draft (and post-draft, flagged) college season, find the
ESPN athlete on his school's roster that season: exact normalized name first, then a
unique fuzzy match (>= 85) on that roster. Rosters come from the box scores of the games
we downloaded, so a player who never appeared in a downloaded game has no link.
"""

from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz

from draft_dna.crosswalk.names import normalize_name

FUZZY_MIN = 85


def link(
    college: pd.DataFrame, team_map: pd.DataFrame, rosters: pd.DataFrame, names: pd.Series
) -> pd.DataFrame:
    """college: bbref_id, season, school_slug, is_pre_draft.
    team_map: season, school_slug, espn_team_id.  rosters: season, team_id, athlete_id,
    athlete_name.  names: bbref_id -> player_name.
    """
    c = college[["bbref_id", "season", "school_slug", "is_pre_draft"]].merge(
        team_map[["season", "school_slug", "espn_team_id"]], on=["season", "school_slug"]
    )
    c["norm"] = c["bbref_id"].map(names).map(normalize_name)
    r = rosters.drop_duplicates(["season", "team_id", "athlete_id"]).copy()
    r["norm"] = r["athlete_name"].map(normalize_name)
    by_team = {k: g for k, g in r.groupby(["season", "team_id"])}
    rows = []
    for row in c.itertuples():
        roster = by_team.get((row.season, row.espn_team_id))
        if roster is None:
            continue
        exact = roster[roster["norm"] == row.norm]
        if len(exact) == 1:
            rows.append(
                (
                    row.bbref_id,
                    row.season,
                    int(exact.iloc[0].athlete_id),
                    row.is_pre_draft,
                    "exact",
                    100.0,
                )
            )
            continue
        scores = roster["norm"].map(lambda n, t=row.norm: fuzz.token_sort_ratio(t, n))
        top = scores.max() if len(scores) else 0
        if top >= FUZZY_MIN and (scores == top).sum() == 1:
            best = roster.loc[scores.idxmax()]
            rows.append(
                (
                    row.bbref_id,
                    row.season,
                    int(best.athlete_id),
                    row.is_pre_draft,
                    "fuzzy",
                    float(top),
                )
            )
    out = pd.DataFrame(
        rows, columns=["bbref_id", "season", "athlete_id", "is_pre_draft", "method", "score"]
    )
    return out.drop_duplicates(["season", "athlete_id"], keep=False)  # ambiguous -> dropped
