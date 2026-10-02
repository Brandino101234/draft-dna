"""stats.nba.com via nba_api: draft history, combine, player directory.

Each call's JSON is cached under data/raw/http/nba_api/ via `cached_json`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    commonplayerinfo,
    draftcombinestats,
    drafthistory,
)

from draft_dna.config import Settings
from draft_dna.ingest.fetcher import cached_json

SOURCE = "nba_api"
TIMEOUT = 90


def result_frame(payload: dict[str, Any], index: int = 0) -> pd.DataFrame:
    rs = payload["resultSets"][index]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def draft_history(settings: Settings | None = None) -> pd.DataFrame:
    payload = cached_json(
        SOURCE,
        "drafthistory/all",
        lambda: drafthistory.DraftHistory(league_id="00", timeout=TIMEOUT).get_dict(),
        settings=settings,
    )
    return result_frame(payload)


def combine(settings: Settings | None = None) -> pd.DataFrame:
    """Anthropometrics, athletic drills and shooting drills, 2000 onward."""
    payload = cached_json(
        SOURCE,
        "draftcombinestats/all_time",
        lambda: draftcombinestats.DraftCombineStats(
            season_all_time="All Time", timeout=TIMEOUT
        ).get_dict(),
        settings=settings,
    )
    return result_frame(payload)


def all_players(settings: Settings | None = None) -> pd.DataFrame:
    """Every NBA player with stats.nba.com person ID and first/last season."""
    payload = cached_json(
        SOURCE,
        "commonallplayers/all",
        lambda: commonallplayers.CommonAllPlayers(
            is_only_current_season=0, timeout=TIMEOUT
        ).get_dict(),
        settings=settings,
    )
    return result_frame(payload)


def player_birth_date(person_id: int, settings: Settings | None = None) -> str | None:
    """Birthdate (YYYY-MM-DD) from CommonPlayerInfo; used to split same-name players."""
    payload = cached_json(
        SOURCE,
        f"commonplayerinfo/{person_id}",
        lambda: commonplayerinfo.CommonPlayerInfo(player_id=person_id, timeout=TIMEOUT).get_dict(),
        settings=settings,
    )
    info = result_frame(payload)
    if info.empty or not info.loc[0, "BIRTHDATE"]:
        return None
    return str(info.loc[0, "BIRTHDATE"])[:10]
