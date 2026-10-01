"""Stage Basketball-Reference raw tables into typed tables."""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.staging.clean import height_to_inches, to_num

MULTI_TEAM = r"^(\dTM|TOT)$"
ID_COLS = {"season", "phase", "name_display", "name_display__id", "team_name_abbr", "pos"}
DROP_SUFFIXES = ("__href",)


def _raw(name: str, s: Settings) -> pd.DataFrame:
    return read_table("raw", "bbref", name, s)


def stage_draft(s: Settings) -> pd.DataFrame:
    d = _raw("draft", s)
    d = d[d["player_name"].notna() & (d["player_name"] != "")].copy()  # forfeited picks
    nh = read_table("raw", "nba_api", "draft_history", s)
    nh = nh.assign(draft_year=nh["SEASON"].astype(int), pick_overall=nh["OVERALL_PICK"].astype(int))
    nh = nh[
        [
            "draft_year",
            "pick_overall",
            "PERSON_ID",
            "PLAYER_NAME",
            "ORGANIZATION",
            "ORGANIZATION_TYPE",
        ]
    ]
    out = d.merge(nh, on=["draft_year", "pick_overall"], how="left", validate="one_to_one")
    out = out.rename(
        columns={
            "PERSON_ID": "nba_person_id",
            "PLAYER_NAME": "nba_player_name",
            "ORGANIZATION": "pre_draft_org",
            "ORGANIZATION_TYPE": "pre_draft_org_type",
        }
    )
    out["prospect_source"] = (
        out["pre_draft_org_type"]
        .map(
            {
                "College/University": "college",
                "High School": "high_school",
                "Other Team/Club": "other_team",
            }
        )
        .fillna("unknown")
    )
    out["nba_person_id"] = out["nba_person_id"].astype("Int64")
    return out.reset_index(drop=True)


def stage_player_index(s: Settings) -> pd.DataFrame:
    p = _raw("player_index", s).rename(columns={"player__id": "bbref_id", "player": "player_name"})
    p["player_name"] = p["player_name"].str.rstrip("*")  # '*' marks Hall of Famers
    p["first_season"] = to_num(p["year_min"]).astype("Int64")
    p["last_season"] = to_num(p["year_max"]).astype("Int64")
    p["height_in"] = height_to_inches(p["height"])
    p["weight_lb"] = to_num(p["weight"])
    p["birth_date"] = pd.to_datetime(p["birth_date"], format="%B %d, %Y", errors="coerce")
    return p[["bbref_id", "player_name", "first_season", "last_season", "pos", "height_in",
              "weight_lb", "birth_date", "colleges"]]  # fmt: skip


def stage_player_bios(s: Settings) -> pd.DataFrame:
    b = _raw("player_bios", s)
    b["birth_date"] = pd.to_datetime(b["birth_date"], errors="coerce")
    b["nba_debut"] = pd.to_datetime(b["nba_debut"], format="%B %d, %Y", errors="coerce")
    for c in ("height_in", "weight_lb", "height_cm", "weight_kg", "recruit_year", "recruit_rank"):
        b[c] = to_num(b[c]).astype("Int64")
    return b


def _season_table(name: str, s: Settings) -> pd.DataFrame:
    df = _raw(f"season_{name}", s)
    df = df[df["name_display__id"].notna()].copy()  # 'League Average' row
    keep = [
        c for c in df.columns if not c.endswith(DROP_SUFFIXES) and c not in {"ranker", "awards"}
    ]
    df = df[keep + (["awards"] if "awards" in df else [])]
    for c in df.columns:
        if c not in ID_COLS and c != "awards":
            df[c] = to_num(df[c])
    return df


def stage_nba_seasons(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player_seasons, team_stints).

    player_seasons: one row per player x season x phase (multi-team players use the
    combined row). team_stints: one row per player x season x phase x team.
    """
    totals = _season_table("totals", s)
    adv = _season_table("advanced", s)
    per100 = _season_table("per_poss", s)
    key = ["season", "phase", "name_display__id", "team_name_abbr"]
    shared = {"name_display", "age", "pos", "games", "games_started", "mp", "awards"}
    adv = adv.drop(columns=[c for c in adv.columns if c in shared])
    per100 = per100.drop(columns=[c for c in per100.columns if c in shared])
    per100 = per100.rename(columns={c: f"{c}_per100" for c in per100.columns if c not in key})
    df = totals.merge(adv, on=key, how="left").merge(per100, on=key, how="left")
    df = df.rename(columns={"name_display__id": "bbref_id", "name_display": "player_name",
                            "team_name_abbr": "team"})  # fmt: skip
    df["is_combined"] = df["team"].str.match(MULTI_TEAM)
    n_teams = df[~df["is_combined"]].groupby(["bbref_id", "season", "phase"]).size()
    df = df.join(n_teams.rename("n_teams"), on=["bbref_id", "season", "phase"])
    stints = df[~df["is_combined"]].drop(columns=["is_combined"]).reset_index(drop=True)
    seasons = df[df["is_combined"] | (df["n_teams"] == 1)].drop(columns=["is_combined"])
    # For multi-team seasons, record the last team (BBRef lists stints chronologically).
    last_team = stints.groupby(["bbref_id", "season", "phase"])["team"].last()
    seasons = seasons.drop(columns=["team"]).join(
        last_team.rename("last_team"), on=["bbref_id", "season", "phase"]
    )
    return seasons.reset_index(drop=True), stints


def stage_team_seasons(s: Settings) -> pd.DataFrame:
    t = _raw("team_seasons", s)
    t["made_playoffs"] = t["team"].str.endswith("*")
    t["team_name"] = t["team"].str.rstrip("*")
    t["team"] = t["team__href"].str.extract(r"/teams/(\w+)/")[0]
    num = ["age", "wins", "losses", "wins_pyth", "losses_pyth", "mov", "sos", "srs",
           "off_rtg", "def_rtg", "net_rtg", "pace"]  # fmt: skip
    for c in num:
        t[c] = to_num(t[c])
    return t[["season", "team", "team_name", "made_playoffs", *num]]


def stage_coaches(s: Settings) -> pd.DataFrame:
    c = _raw("coaches", s)
    c = c[c["coach"].notna() & (c["coach"] != "")].copy()
    c["coach_id"] = c["coach__href"].str.extract(r"/coaches/(\w+)\.html")[0]
    out = c[["season", "team", "coach", "coach_id"]].copy()
    out["season_num_with_team"] = to_num(c["seas_num_franch"]).astype("Int64")
    out["games_coached"] = to_num(c["cur_g"]).astype("Int64")
    out["wins"] = to_num(c["cur_w"]).astype("Int64")
    return out.reset_index(drop=True)


def stage_awards(s: Settings) -> pd.DataFrame:
    """Long table: season, award, bbref_id, rank/team, vote share, selected flag."""
    a = _raw("awards", s).rename(columns={"player__id": "bbref_id"})
    a = a[a["bbref_id"].notna()].copy()
    a["award_share"] = to_num(a.get("award_share", pd.Series(index=a.index, dtype=str)))
    team_col = a.get("all_nba_team", pd.Series(index=a.index, dtype="string"))
    for c in ("all_defense_team", "all_rookie_team"):
        if c in a:
            team_col = team_col.fillna(a[c])
    a["team_level"] = team_col.replace("", pd.NA)
    a["vote_rank"] = to_num(a["rank"]).astype("Int64") if "rank" in a else pd.NA
    team_awards = a["award"].isin(["all_nba", "all_defense", "all_rookie"])
    a["selected"] = (team_awards & a["team_level"].isin(["1st", "2nd", "3rd"])) | (
        ~team_awards & (a["vote_rank"] == 1)
    )
    return a[["season", "award", "bbref_id", "team_level", "vote_rank", "award_share", "selected"]]


def stage_all_stars(s: Settings) -> pd.DataFrame:
    a = _raw("all_stars", s).rename(columns={"player__id": "bbref_id"})
    return a[["season", "bbref_id", "all_star_team"]].drop_duplicates(["season", "bbref_id"])


def stage_league_averages(s: Settings) -> pd.DataFrame:
    la = _raw("league_averages", s)
    la["season"] = to_num(la["season"].str.slice(0, 4)).astype("Int64") + 1
    cols = ["pace", "off_rtg", "ts_pct", "efg_pct", "fg3a_per_g", "fga_per_g", "fta_per_g",
            "pts_per_g", "tov_pct", "orb_pct", "ft_rate"]  # fmt: skip
    for c in cols:
        la[c] = to_num(la[c])
    la["fg3a_rate"] = la["fg3a_per_g"] / la["fga_per_g"]
    return la[["season", *cols, "fg3a_rate"]].dropna(subset=["pace"])


def stage_player_salaries(s: Settings) -> pd.DataFrame:
    sal = _raw("player_salaries", s)
    sal["season"] = to_num(sal["season"].str.slice(0, 4)).astype("Int64") + 1
    sal["salary"] = to_num(sal["salary"])
    sal["team"] = sal["team_name__href"].str.extract(r"/teams/(\w+)/")[0]
    return sal[["bbref_id", "season", "team", "salary"]].dropna(subset=["salary"])


def run(s: Settings) -> None:
    write_table(stage_draft(s), "staging", "bbref", "draft_picks", s)
    write_table(stage_player_index(s), "staging", "bbref", "player_index", s)
    seasons, stints = stage_nba_seasons(s)
    write_table(seasons, "staging", "bbref", "nba_player_seasons", s)
    write_table(stints, "staging", "bbref", "nba_team_stints", s)
    write_table(stage_team_seasons(s), "staging", "bbref", "nba_team_seasons", s)
    write_table(stage_coaches(s), "staging", "bbref", "nba_coaches", s)
    write_table(stage_awards(s), "staging", "bbref", "nba_awards", s)
    write_table(stage_all_stars(s), "staging", "bbref", "nba_all_stars", s)
    write_table(stage_league_averages(s), "staging", "bbref", "nba_league_averages", s)
    if table_path("raw", "bbref", "player_bios", s).exists():
        write_table(stage_player_bios(s), "staging", "bbref", "player_bios", s)
        write_table(stage_player_salaries(s), "staging", "bbref", "player_salaries", s)
