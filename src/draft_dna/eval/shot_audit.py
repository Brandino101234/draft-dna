"""Phase 4a: audit college shot-data coverage before any modeling.

A stratified sample: for every college season 2008-2026, SAMPLE_TEAMS random team-seasons
that produced a draft pick, and SAMPLE_GAMES random games from each. For every sampled
game, every field-goal attempt is checked for (a) a shot type (layup / dunk / jumper /
three / tip) and (b) valid x/y coordinates.

Coverage is then summarized by season, conference, broadcast and drafted player, and the
cost of a full pull is estimated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.crosswalk.names import normalize_name
from draft_dna.ingest import espn
from draft_dna.ingest.fetcher import Fetcher
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

SEASONS = range(2008, 2027)
SAMPLE_TEAMS = 20
SAMPLE_GAMES = 8
SEED = 0
COORD_GAME_SHARE = 0.9  # a game "has coordinates" when >=90% of its FGA do


def classify(shot_type: str, text: str, score_value: float | None) -> str:
    t = f"{shot_type} {text}".lower()
    if "dunk" in t:
        return "dunk"
    if "tip" in t:
        return "tip"
    if "layup" in t or "lay up" in t:
        return "layup"
    if "three" in t or "3-pt" in t or score_value == 3:
        return "three"
    if "jump" in t or "jumper" in t:
        return "jumper"
    return "unknown"


def team_seasons(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team-seasons of drafted players' pre-draft college seasons, with ESPN team ids."""
    c = read_table("modeled", "core", "college_player_seasons", s)
    p = read_table("modeled", "core", "players", s).set_index("bbref_id")
    c = c[c["is_pre_draft"] & c["season"].isin(list(SEASONS)) & c["school_slug"].notna()]
    c = c[c["bbref_id"].map(p["drafted"]).fillna(False).astype(bool)]
    names = read_table("staging", "cbb", "team_seasons", s).drop_duplicates(
        "school_slug", keep="last"
    )
    names = names.set_index("school_slug")["school_name"].to_dict()
    f = Fetcher(espn.SOURCE, settings=s)
    mapping = espn.map_schools(sorted(c["school_slug"].unique()), names, espn.teams(f))
    ts = c[["season", "school_slug", "conf_abbr"]].drop_duplicates(["season", "school_slug"])
    ts = ts.merge(mapping[["school_slug", "espn_team_id"]], on="school_slug", how="left")
    players = c[["bbref_id", "season", "school_slug", "fga"]]
    return ts, players


def pull_sample(s: Settings) -> None:
    ts, players = team_seasons(s)
    write_table(players, "raw", "espn", "audit_drafted_player_seasons", s)
    f = Fetcher(espn.SOURCE, settings=s)
    rng = np.random.default_rng(SEED)
    sampled, games, shots, rosters, infos = [], [], [], [], []
    for season in SEASONS:
        pool = ts[(ts["season"] == season) & ts["espn_team_id"].notna()]
        pick = pool.iloc[rng.permutation(len(pool))[:SAMPLE_TEAMS]]
        for r in pick.itertuples():
            sched = espn.schedule(f, int(r.espn_team_id), season)
            if sched.empty:
                continue
            past = sched[pd.to_datetime(sched["date"], utc=True) < pd.Timestamp.now(tz="UTC")]
            chosen = past.iloc[rng.permutation(len(past))[:SAMPLE_GAMES]].assign(
                school_slug=r.school_slug, conf=r.conf_abbr
            )
            sampled.append(r)
            games.append(chosen)
            for gid in chosen["game_id"]:
                sh, ro, info = espn.game_shots(f, int(gid))
                shots.append(sh.assign(sample_team_id=int(r.espn_team_id), season=season))
                rosters.append(ro.assign(season=season))
                infos.append(info)
        log.info(
            "audit sample: season %d done (%d network requests so far)", season, f.network_requests
        )
    write_table(pd.concat(games, ignore_index=True), "raw", "espn", "audit_games", s)
    write_table(pd.concat(shots, ignore_index=True), "raw", "espn", "audit_shots", s)
    write_table(pd.concat(rosters, ignore_index=True), "raw", "espn", "audit_rosters", s)
    write_table(pd.DataFrame(infos), "raw", "espn", "audit_game_info", s)
    write_table(ts, "raw", "espn", "audit_team_seasons", s)


def analyze(s: Settings) -> dict[str, pd.DataFrame]:
    games = read_table("raw", "espn", "audit_games", s)
    shots = read_table("raw", "espn", "audit_shots", s)
    info = read_table("raw", "espn", "audit_game_info", s)
    shots["kind"] = [
        classify(t, x or "", v)
        for t, x, v in zip(shots["shot_type"], shots["text"], shots["score_value"], strict=True)
    ]
    shots["has_xy"] = shots["x"].notna()
    # Only the sampled team's attempts (opponents vary in coverage independently).
    own = shots[shots["team_id"] == shots["sample_team_id"]]
    per_game = (
        own.groupby(["season", "game_id"])
        .agg(
            fga=("has_xy", "size"),
            xy_share=("has_xy", "mean"),
            typed_share=("kind", lambda k: (k != "unknown").mean()),
            assisted_share=("assist_athlete_id", lambda a: a.notna().mean()),
        )
        .reset_index()
    )
    per_game["has_xy"] = per_game["xy_share"] >= COORD_GAME_SHARE
    g = games.drop_duplicates(["game_id", "school_slug"]).merge(
        per_game, on=["season", "game_id"], how="left"
    )
    g = g.merge(info[["game_id", "available", "n_plays"]], on="game_id", how="left")
    g["has_pbp"] = g["fga"].fillna(0) > 20
    g["televised"] = g["broadcast"].notna()

    by_season = (
        g.groupby("season")
        .agg(
            games=("game_id", "size"),
            pbp=("has_pbp", "mean"),
            xy_games=("has_xy", lambda x: x.fillna(False).mean()),
            typed=("typed_share", "mean"),
        )
        .reset_index()
    )
    sh_season = (
        own.groupby("season")
        .agg(
            fga=("has_xy", "size"),
            fga_with_xy=("has_xy", "mean"),
            fga_typed=("kind", lambda k: (k != "unknown").mean()),
        )
        .reset_index()
    )
    by_season = by_season.merge(sh_season, on="season")
    by_conf = (
        g[g["has_pbp"]]
        .groupby("conf")
        .agg(games=("game_id", "size"), xy_games=("has_xy", lambda x: x.fillna(False).mean()))
        .sort_values("games", ascending=False)
        .reset_index()
    )
    by_tv = (
        g[g["has_pbp"] & (g["season"] >= 2015)]
        .groupby("televised")
        .agg(games=("game_id", "size"), xy_games=("has_xy", lambda x: x.fillna(False).mean()))
        .reset_index()
    )
    kinds = own.groupby("season")["kind"].value_counts(normalize=True).unstack().fillna(0)

    players = player_coverage(s, own)
    return {
        "games": g,
        "by_season": by_season,
        "by_conf": by_conf,
        "by_tv": by_tv,
        "kinds": kinds.reset_index(),
        "players": players,
    }


def player_coverage(s: Settings, own: pd.DataFrame) -> pd.DataFrame:
    """Share of each sampled drafted player's FGA that carry coordinates."""
    rosters = read_table("raw", "espn", "audit_rosters", s).drop_duplicates(
        ["season", "athlete_id"]
    )
    ps = read_table("raw", "espn", "audit_drafted_player_seasons", s)
    ts = read_table("raw", "espn", "audit_team_seasons", s)
    names = read_table("modeled", "core", "players", s).set_index("bbref_id")["player_name"]
    ps = ps.merge(ts[["season", "school_slug", "espn_team_id"]], on=["season", "school_slug"])
    ps["norm"] = ps["bbref_id"].map(names).map(normalize_name)
    rosters["norm"] = rosters["athlete_name"].map(normalize_name)
    m = ps.merge(
        rosters,
        left_on=["season", "espn_team_id", "norm"],
        right_on=["season", "team_id", "norm"],
        how="inner",
    )
    sampled_teams = own[["season", "sample_team_id"]].drop_duplicates()
    m = m.merge(
        sampled_teams, left_on=["season", "espn_team_id"], right_on=["season", "sample_team_id"]
    )
    shot = (
        own.groupby(["season", "athlete_id"])
        .agg(sample_fga=("has_xy", "size"), xy_share=("has_xy", "mean"))
        .reset_index()
    )
    out = m[["bbref_id", "season", "school_slug", "athlete_id", "fga"]].merge(
        shot, on=["season", "athlete_id"], how="left"
    )
    out["sample_fga"] = out["sample_fga"].fillna(0)
    return out


def full_pull_estimate(s: Settings) -> dict[str, float]:
    ts = read_table("raw", "espn", "audit_team_seasons", s)
    games = read_table("raw", "espn", "audit_games", s)
    per_team = games.groupby(["season", "school_slug"]).size()
    n_ts = int(ts["espn_team_id"].notna().sum())
    est_games = n_ts * 32
    secs = est_games * s.rate_limits["espn"].min_seconds_between_requests * 1.15
    return {
        "team_seasons": n_ts,
        "sampled_team_seasons": len(per_team),
        "estimated_games": est_games,
        "estimated_hours": round(secs / 3600, 1),
    }


def sample_exists(s: Settings) -> bool:
    return table_path("raw", "espn", "audit_shots", s).exists()
