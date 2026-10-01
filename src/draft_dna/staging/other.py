"""Stage nba_api, Barttorvik and Sports-Reference CBB raw tables."""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.staging.clean import height_to_inches, season_end_year, to_num

# ----------------------------------------------------------------------------- nba_api
COMBINE_NUMERIC = [
    "HEIGHT_WO_SHOES", "HEIGHT_W_SHOES", "WEIGHT", "WINGSPAN", "STANDING_REACH",
    "BODY_FAT_PCT", "HAND_LENGTH", "HAND_WIDTH", "STANDING_VERTICAL_LEAP",
    "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "MODIFIED_LANE_AGILITY_TIME",
    "THREE_QUARTER_SPRINT", "BENCH_PRESS",
]  # fmt: skip


def stage_combine(s: Settings) -> pd.DataFrame:
    c = read_table("raw", "nba_api", "combine", s)
    out = pd.DataFrame(
        {
            "combine_year": c["SEASON"].astype(int),
            "nba_person_id": c["PLAYER_ID"].astype("Int64"),
            "player_name": c["PLAYER_NAME"],
            "combine_position": c["POSITION"],
        }
    )
    for col in COMBINE_NUMERIC:
        out[col.lower()] = to_num(c[col])
    # A handful of rows carry placeholder IDs (-1 or out of range): keep them, unlinked.
    valid = out["nba_person_id"].between(1, 10**8)
    out["nba_person_id"] = out["nba_person_id"].where(valid)
    # Players who attended twice (withdrew, returned) keep one row per combine year;
    # features pick the latest combine on or before the draft year.
    return out.drop_duplicates(["nba_person_id", "combine_year", "player_name"])


def stage_nba_players(s: Settings) -> pd.DataFrame:
    p = read_table("raw", "nba_api", "all_players", s)
    return pd.DataFrame(
        {
            "nba_person_id": p["PERSON_ID"].astype("Int64"),
            "player_name": p["DISPLAY_FIRST_LAST"],
            # FROM_YEAR is the season's starting year; convert to ending year like BBRef.
            "first_season": to_num(p["FROM_YEAR"]).astype("Int64") + 1,
            "last_season": to_num(p["TO_YEAR"]).astype("Int64") + 1,
            "played": p["GAMES_PLAYED_FLAG"].eq("Y"),
        }
    )


# --------------------------------------------------------------------------- barttorvik
BART_TEXT = {"player_name", "team", "conf", "class_yr", "height", "jersey", "hometown",
             "role", "birth_date", "pid", "year"}  # fmt: skip


def stage_barttorvik_players(s: Settings) -> pd.DataFrame:
    b = read_table("raw", "barttorvik", "players", s)
    out = b.copy()
    for c in out.columns:
        if c not in BART_TEXT:
            out[c] = to_num(out[c])
    out["season"] = out["year"].astype(int)
    out["bart_pid"] = out["pid"].astype(int)
    out["height_in"] = height_to_inches(out["height"])
    bd = pd.to_datetime(out["birth_date"].replace("", pd.NA), errors="coerce")
    # Unknown birthdates are filled with October 15 of an estimated year; treat as missing.
    placeholder = (bd.dt.month == 10) & (bd.dt.day == 15)
    out["birth_date"] = bd.where(~placeholder)
    out["nba_pick"] = out["nba_pick"].astype("Int64")
    return out.drop(columns=["year", "pid", "height"])


def stage_barttorvik_teams(s: Settings) -> pd.DataFrame:
    t = read_table("raw", "barttorvik", "teams", s)
    keep = {"year": "season", "team": "team", "conf": "conf", "adjoe": "adj_oe",
            "adjde": "adj_de", "barthag": "barthag", "sos": "sos", "adjt": "adj_tempo",
            "wab": "wab"}  # fmt: skip
    out = t[list(keep)].rename(columns=keep)
    for c in ("adj_oe", "adj_de", "barthag", "sos", "adj_tempo", "wab"):
        out[c] = to_num(out[c])
    return out


# ---------------------------------------------------------------------------------- cbb
CBB_TEXT = {"cbb_id", "year_id", "team_name_abbr", "conf_abbr", "class", "pos", "awards"}


def _cbb_player_table(name: str, s: Settings) -> pd.DataFrame:
    df = read_table("raw", "cbb", f"player_{name}", s)
    df = df[[c for c in df.columns if not c.endswith("__href") or c == "team_name_abbr__href"]]
    for c in df.columns:
        if c not in CBB_TEXT and c != "team_name_abbr__href":
            df[c] = to_num(df[c])
    df["season"] = season_end_year(df["year_id"]).astype("Int64")
    df["school_slug"] = df["team_name_abbr__href"].str.extract(r"/cbb/schools/([\w-]+)/")[0]
    return df.drop(columns=["team_name_abbr__href"])


def stage_cbb_player_seasons(s: Settings) -> pd.DataFrame:
    """One row per player x season x school (transfers produce multiple rows)."""
    key = ["cbb_id", "season", "school_slug"]
    totals = _cbb_player_table("totals", s)
    adv = _cbb_player_table("advanced", s)
    adv = adv[[c for c in adv.columns if c not in totals.columns or c in key]]
    out = totals.merge(adv, on=key, how="left")
    if table_path("raw", "cbb", "player_per_poss", s).exists():
        pp = _cbb_player_table("per_poss", s)
        pp = pp[[c for c in pp.columns if c not in out.columns or c in key]]
        pp = pp.rename(columns={c: f"{c}_per100" for c in pp.columns if c not in key})
        out = out.merge(pp, on=key, how="left")
    return out.drop_duplicates(key)


def stage_cbb_team_seasons(s: Settings) -> pd.DataFrame:
    basic = read_table("raw", "cbb", "team_school_stats", s)
    basic["school_slug"] = basic["school_name__href"].str.extract(r"/cbb/schools/([\w-]+)/")[0]
    num = ["g", "wins", "losses", "srs", "sos", "pts", "opp_pts", "mp", "fg", "fga", "fg3",
           "fg3a", "ft", "fta", "orb", "trb", "ast", "stl", "blk", "tov", "pf"]  # fmt: skip
    out = basic[["season", "school_slug", "school_name"]].copy()
    for c in num:
        out[c] = to_num(basic[c])
    if table_path("raw", "cbb", "team_advanced_school_stats", s).exists():
        adv = read_table("raw", "cbb", "team_advanced_school_stats", s)
        adv["school_slug"] = adv["school_name__href"].str.extract(r"/cbb/schools/([\w-]+)/")[0]
        acols = ["pace", "off_rtg", "trb_pct", "ast_pct", "stl_pct", "blk_pct", "tov_pct",
                 "orb_pct", "ts_pct", "fg3a_per_fga_pct"]  # fmt: skip
        adv = adv[["season", "school_slug", *acols]].copy()
        for c in acols:
            adv[c] = to_num(adv[c])
        adv = adv.rename(columns={c: f"team_{c}" for c in acols})
        out = out.merge(adv, on=["season", "school_slug"], how="left")
    out["school_name"] = out["school_name"].str.replace(r"\s*NCAA$", "", regex=True)
    return out.drop_duplicates(["season", "school_slug"])


def run(s: Settings) -> None:
    write_table(stage_combine(s), "staging", "nba_api", "combine", s)
    write_table(stage_nba_players(s), "staging", "nba_api", "players", s)
    write_table(stage_barttorvik_players(s), "staging", "barttorvik", "player_seasons", s)
    write_table(stage_barttorvik_teams(s), "staging", "barttorvik", "team_seasons", s)
    if table_path("raw", "cbb", "player_totals", s).exists():
        write_table(stage_cbb_player_seasons(s), "staging", "cbb", "player_seasons", s)
    if table_path("raw", "cbb", "team_school_stats", s).exists():
        write_table(stage_cbb_team_seasons(s), "staging", "cbb", "team_seasons", s)
