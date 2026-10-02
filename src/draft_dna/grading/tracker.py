"""Live tracker for the newest draft class: projection vs reality during the season.

For each rookie: the draft-night projected range of rookie-season value (from the same
draft-slot model of record, horizon 1), and once games are played, his season-to-date
value and full-season pace (value scaled by team games remaining). Updated by
`draft-dna refresh`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, table_path


def rookie_tracker(s: Settings, draft_year: int | None = None) -> pd.DataFrame:
    draft_year = draft_year or s.draft_classes.live[-1]
    season = draft_year + 1
    players = read_table("modeled", "core", "players", s)
    rookies = players[players["drafted"] & (players["draft_year"] == draft_year)]
    out = (
        rookies[["bbref_id", "player_name", "pick_overall", "team_id"]]
        .rename(columns={"pick_overall": "pick", "team_id": "team"})
        .set_index("bbref_id")
    )
    bands = read_table("modeled", "grading", "trajectory_bands", s)
    b1 = bands[bands["n"] == 1].set_index("bbref_id")
    # Year-1 peak band is (season value) / 3 by construction; express it as season value.
    for col in ("floor", "median", "ceiling"):
        out[f"projected_{col}"] = b1[col].reindex(out.index) * 3

    out["games"] = 0
    out["minutes"] = 0.0
    out["value_to_date"] = np.nan
    out["value_pace"] = np.nan
    seasons = read_table("modeled", "core", "nba_player_seasons", s)
    current = seasons[(seasons["season"] == season) & seasons["bbref_id"].isin(out.index)]
    if not current.empty and table_path("modeled", "outcomes", "season_values", s).exists():
        values = read_table("modeled", "outcomes", "season_values", s)
        v = values[values["season"] == season].set_index("bbref_id")["value_blend"]
        cur = current.set_index("bbref_id")
        teams = read_table("staging", "bbref", "nba_team_seasons", s)
        played = teams[teams["season"] == season].assign(g=lambda d: d["wins"] + d["losses"])
        team_g = played.set_index("team")["g"]
        out.loc[cur.index, "games"] = cur["games"]
        out.loc[cur.index, "minutes"] = cur["mp"]
        out["value_to_date"] = v.reindex(out.index)
        g = cur["last_team"].map(team_g).reindex(out.index)
        out["value_pace"] = out["value_to_date"] * 82 / g.where(g > 0)
    out["status"] = np.select(
        [
            (out["games"] == 0).to_numpy(dtype=bool),
            (out["value_pace"] > out["projected_ceiling"]).fillna(False).to_numpy(dtype=bool),
            (out["value_pace"] < out["projected_floor"]).fillna(False).to_numpy(dtype=bool),
        ],
        ["no games yet", "pacing above ceiling", "pacing below floor"],
        "within range",
    )
    return out.reset_index().sort_values("pick")
