"""Integrated (modeled-layer) tables keyed on bbref_id.

These are the inputs for Phase 2+ (outcomes, features). No outcome definitions or
model features live here, only joined, cleaned facts.
"""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings
from draft_dna.crosswalk.build import normalize_school, school_sim
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

AWARD_FLAGS = {
    "all_nba": "all_nba",
    "all_defense": "all_defense",
    "all_rookie": "all_rookie",
    "mvp": "mvp",
    "roy": "roy",
    "dpoy": "dpoy",
    "smoy": "smoy",
    "mip": "mip",
}


def players(s: Settings) -> pd.DataFrame:
    xw = read_table("modeled", "xwalk", "players", s)
    xw["draft_class_role"] = pd.cut(
        xw["draft_year"],
        bins=[0, s.draft_classes.training[1], s.draft_classes.in_progress[1], 9999],
        labels=["training", "in_progress", "live"],
    ).astype("string")
    xw.loc[~xw["drafted"], "draft_class_role"] = "undrafted"
    return xw


def nba_player_seasons(s: Settings, universe: pd.DataFrame) -> pd.DataFrame:
    ps = read_table("staging", "bbref", "nba_player_seasons", s)
    ps = ps[ps["bbref_id"].isin(universe["bbref_id"])]
    reg = ps[ps["phase"] == "regular"].drop(columns=["phase"])
    post = ps[ps["phase"] == "playoffs"]
    post_cols = ["games", "mp", "pts", "ws", "bpm", "vorp", "obpm", "dbpm", "ts_pct"]
    post = post[["bbref_id", "season", *[c for c in post_cols if c in post]]]
    post = post.rename(columns={c: f"po_{c}" for c in post_cols})
    out = reg.merge(post, on=["bbref_id", "season"], how="left")
    out["made_playoffs_roster"] = out["po_games"].notna()

    # Season length (82, but 50 in 1999, 66 in 2012, 72 in 2021, and uneven in 2020).
    teams = read_table("staging", "bbref", "nba_team_seasons", s)
    teams["team_games"] = teams["wins"] + teams["losses"]
    season_len = teams.groupby("season")["team_games"].max()
    out["season_games"] = out["season"].map(season_len)
    # Absences include injuries, rest, coach's decisions and time in the G League.
    out["games_absent"] = (out["season_games"] - out["games"]).clip(lower=0)

    # Team context for the player's last team that season.
    tq = teams.rename(columns={"team": "last_team"})[
        ["season", "last_team", "wins", "losses", "srs", "pace", "made_playoffs"]
    ].rename(columns={"wins": "team_wins", "losses": "team_losses", "srs": "team_srs",
                      "pace": "team_pace", "made_playoffs": "team_made_playoffs"})  # fmt: skip
    out = out.merge(tq, on=["season", "last_team"], how="left")
    coaches = read_table("staging", "bbref", "nba_coaches", s)
    n_coaches = coaches.groupby(["season", "team"]).size().rename("team_coaches_in_season")
    head = coaches.sort_values("games_coached").groupby(["season", "team"]).last()
    out = out.join(n_coaches, on=["season", "last_team"])
    out = out.join(
        head[["coach_id", "season_num_with_team"]].rename(
            columns={"coach_id": "head_coach_id", "season_num_with_team": "coach_season_with_team"}
        ),
        on=["season", "last_team"],
    )

    # Awards.
    aw = read_table("staging", "bbref", "nba_awards", s)
    sel = aw[aw["selected"]]
    for award, col in AWARD_FLAGS.items():
        a = sel[sel["award"] == award]
        if award in ("all_nba", "all_defense", "all_rookie"):
            lvl = a.set_index(["bbref_id", "season"])["team_level"]
            out = out.join(lvl.rename(f"{col}_team"), on=["bbref_id", "season"])
        else:
            keys = set(zip(a["bbref_id"], a["season"], strict=True))
            out[f"won_{col}"] = [
                k in keys for k in zip(out["bbref_id"], out["season"], strict=True)
            ]
    mvp = aw[aw["award"] == "mvp"].set_index(["bbref_id", "season"])["award_share"]
    out = out.join(mvp.rename("mvp_share"), on=["bbref_id", "season"])
    stars = read_table("staging", "bbref", "nba_all_stars", s)
    star_keys = set(zip(stars["bbref_id"], stars["season"], strict=True))
    out["all_star"] = [k in star_keys for k in zip(out["bbref_id"], out["season"], strict=True)]

    # Season number relative to draft (season 1 = the season after the draft).
    u = universe.set_index("bbref_id")
    base_year = u["draft_year"].where(u["drafted"], u["first_season"] - 1)
    out["season_num"] = out["season"] - out["bbref_id"].map(base_year)
    # Count of seasons actually played up to and including this one.
    out = out.sort_values(["bbref_id", "season"])
    out["nba_season_index"] = out.groupby("bbref_id").cumcount() + 1
    return out.reset_index(drop=True)


def college_player_seasons(s: Settings, universe: pd.DataFrame) -> pd.DataFrame:
    """Sports-Reference college seasons with Barttorvik advanced stats and SOS."""
    cbb = read_table("staging", "cbb", "player_seasons", s)
    ids = universe[["bbref_id", "cbb_id", "bart_pid"]].dropna(subset=["cbb_id"])
    out = cbb.merge(ids, on="cbb_id", how="inner")

    teams = read_table("staging", "cbb", "team_seasons", s)
    team_cols = ["season", "school_slug", "srs", "sos", "wins", "losses", "team_pace"]
    tc = teams[[c for c in team_cols if c in teams]].rename(
        columns={"srs": "team_srs", "sos": "team_sos", "wins": "team_wins", "losses": "team_losses"}
    )
    out = out.merge(tc, on=["season", "school_slug"], how="left")

    bart = read_table("staging", "barttorvik", "player_seasons", s)
    bart_cols = ["bart_pid", "season", "team", "usg", "ortg", "drtg", "bpm", "obpm", "dbpm",
                 "orb_pct", "drb_pct", "ast_pct", "tov_pct", "blk_pct", "stl_pct", "ftr",
                 "rim_made", "rim_att", "mid_made", "mid_att", "dunks_made", "dunks_att",
                 "three_pm", "three_pa", "recruit_rating", "porpag", "min_pct"]  # fmt: skip
    b = bart[bart_cols].rename(
        columns={c: f"bart_{c}" for c in bart_cols if c not in ("bart_pid", "season")}
    )
    out = out.merge(b, on=["bart_pid", "season"], how="left")
    # A transfer's season can match the wrong Barttorvik row (two schools); keep the
    # Barttorvik numbers only when the school agrees.
    has_bart = out["bart_team"].notna()
    same_school = [
        school_sim(a, b) >= 80 or normalize_school(a) in normalize_school(b)
        for a, b in zip(out["school_name"] if "school_name" in out else out["team_name_abbr"],
                        out["bart_team"].fillna(""), strict=True)
    ]  # fmt: skip
    mismatch = has_bart & ~pd.Series(same_school, index=out.index)
    bart_value_cols = [c for c in out.columns if c.startswith("bart_") and c != "bart_pid"]
    out.loc[mismatch, bart_value_cols] = pd.NA
    out["bart_school_mismatch"] = mismatch
    bteams = read_table("staging", "barttorvik", "team_seasons", s)
    bt = bteams.rename(
        columns={"team": "bart_team", "sos": "bart_team_sos", "barthag": "bart_team_barthag",
                 "adj_tempo": "bart_team_tempo", "conf": "bart_conf"}
    )  # fmt: skip
    out = out.merge(
        bt[
            [
                "season",
                "bart_team",
                "bart_team_sos",
                "bart_team_barthag",
                "bart_team_tempo",
                "bart_conf",
            ]
        ],
        on=["season", "bart_team"],
        how="left",
    )
    return out


def combine(s: Settings) -> pd.DataFrame:
    links = read_table("modeled", "xwalk", "combine_links", s)
    c = read_table("staging", "nba_api", "combine", s)
    by_id = links.merge(
        read_table("modeled", "xwalk", "players", s)[["bbref_id", "nba_person_id"]], on="bbref_id"
    )
    out = by_id.merge(c, on=["nba_person_id", "combine_year"], how="left")
    unmatched = out["height_wo_shoes"].isna() & out["wingspan"].isna() & out["weight"].isna()
    return out[~unmatched | out["player_name"].notna()]


def run(s: Settings) -> None:
    universe = players(s)
    write_table(universe, "modeled", "core", "players", s)
    write_table(nba_player_seasons(s, universe), "modeled", "core", "nba_player_seasons", s)
    write_table(combine(s), "modeled", "core", "combine", s)
    if table_path("staging", "cbb", "player_seasons", s).exists():
        write_table(
            college_player_seasons(s, universe), "modeled", "core", "college_player_seasons", s
        )
