"""Pre-draft feature table: one row per player, using only information available on
draft night.

Leakage rules applied here:
- College seasons must be flagged `is_pre_draft` (drafted players can return to school).
- Weight comes from the combine only (Basketball-Reference lists current NBA weight).
- Height prefers the combine; listed height is a fallback (it barely changes).
- Position comes from the last college season or the combine, not the NBA listing.
- The combine used is the latest one on or before the draft year.
- Nothing here is standardized: scaling happens inside each backtest fold.

Undrafted players get features (for comps and descriptive work) using their final
college season; they are not part of the projection models, which need a pick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

CLASS_YEARS = {"FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 5}
# College listings are mostly G / F / C, so wings fold into forwards.
POSITION_BUCKET = {"G": "guard", "PG": "guard", "SG": "guard", "G-F": "forward", "F-G": "forward",
                   "SF": "forward", "F": "forward", "PF": "forward", "F-C": "big", "C-F": "big",
                   "C": "big"}  # fmt: skip

# Feature groups (used by models and for missingness reporting).
BIO = ["age_at_draft", "height_in", "weight_lb", "wingspan_in", "wingspan_minus_height",
       "standing_reach_in", "max_vertical", "lane_agility", "sprint",
       "recruit_rank_top100"]  # fmt: skip
COLLEGE = ["college_seasons", "games", "mpg", "pts_per40", "trb_per40", "ast_per40",
           "stl_per40", "blk_per40", "tov_per40", "ts_pct", "ts_rel", "fg3a_rate",
           "fg3a_rate_rel", "fg3_pct", "fg3a_per40", "ft_pct", "career_ft_pct", "ftr", "usg_pct",
           "ast_pct", "tov_pct", "trb_pct", "blk_pct", "stl_pct", "bpm", "obpm", "dbpm",
           "team_srs", "team_sos"]  # fmt: skip
SHOT_TYPE = ["rim_rate", "mid_rate", "rim_fg_pct", "dunk_rate"]  # Barttorvik, 2010+
FLAGS = ["src_college", "src_high_school", "src_other_team", "pos_guard", "pos_forward",
         "pos_big"]  # fmt: skip
STATS_FEATURES = BIO + COLLEGE + FLAGS


def _last_pre_draft_season(college: pd.DataFrame) -> pd.DataFrame:
    """The final pre-draft college season per player; for a mid-season transfer, the
    row with the most minutes."""
    c = college[college["is_pre_draft"]].sort_values(["bbref_id", "season", "mp"])
    return c.groupby("bbref_id").tail(1).set_index("bbref_id")


def _career_college(college: pd.DataFrame) -> pd.DataFrame:
    c = college[college["is_pre_draft"]]
    g = c.groupby("bbref_id")
    out = pd.DataFrame(
        {
            "college_seasons": g["season"].nunique(),
            "career_ft_pct": g["ft"].sum() / g["fta"].sum().replace(0, np.nan),
        }
    )
    return out


def build(s: Settings) -> pd.DataFrame:
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    college = read_table("modeled", "core", "college_player_seasons", s)
    combine = read_table("modeled", "core", "combine", s)
    bios = read_table("staging", "bbref", "player_bios", s).set_index("bbref_id")

    f = pd.DataFrame(index=players.index)
    f["draft_year"] = players["draft_year"]
    f["pick"] = players["pick_overall"]
    f["drafted"] = players["drafted"]
    f["prospect_source"] = players["prospect_source"].fillna("unknown")
    ref_year = players["draft_year"].fillna(players["first_season"] - 1)
    draft_day = pd.to_datetime(ref_year.astype("Int64").astype(str) + "-06-25", errors="coerce")
    f["age_at_draft"] = (draft_day - players["birth_date"]).dt.days / 365.25
    f["recruit_rank_top100"] = bios["recruit_rank"].reindex(f.index).astype(float)

    # Combine: latest attendance on or before the draft year.
    cb = combine.join(ref_year.rename("ref_year"), on="bbref_id")
    cb = cb[cb["combine_year"] <= cb["ref_year"]].sort_values("combine_year")
    cb = cb.groupby("bbref_id").tail(1).set_index("bbref_id").reindex(f.index)
    listed_height = players["height_in"].astype(float)
    f["height_in"] = cb["height_wo_shoes"].fillna(listed_height)
    f["height_from_combine"] = cb["height_wo_shoes"].notna()
    f["weight_lb"] = cb["weight"]
    f["wingspan_in"] = cb["wingspan"]
    f["wingspan_minus_height"] = cb["wingspan"] - cb["height_wo_shoes"]
    f["standing_reach_in"] = cb["standing_reach"]
    f["max_vertical"] = cb["max_vertical_leap"]
    f["lane_agility"] = cb["lane_agility_time"]
    f["sprint"] = cb["three_quarter_sprint"]

    # College: final pre-draft season + career aggregates.
    last = _last_pre_draft_season(college).reindex(f.index)
    per40 = 40 / last["mp"].where(last["mp"] > 0)
    f["games"] = last["games"]
    f["mpg"] = last["mp"] / last["games"].where(last["games"] > 0)
    for stat in ("pts", "trb", "ast", "stl", "blk", "tov", "fg3a"):
        f[f"{stat}_per40"] = last[stat] * per40
    f["ts_pct"] = last["ts_pct"]
    f["ts_rel"] = last["ts_rel"]
    f["fg3a_rate"] = last["fg3a"] / last["fga"].where(last["fga"] > 0)
    f["fg3a_rate_rel"] = last["fg3a_rate_rel"]
    f["fg3_pct"] = last["fg3_pct"]
    f["ft_pct"] = last["ft_pct"]
    f["ftr"] = last["fta"] / last["fga"].where(last["fga"] > 0)
    for stat in ("usg_pct", "ast_pct", "tov_pct", "trb_pct", "blk_pct", "stl_pct"):
        f[stat] = last[stat]
    # BPM: Sports-Reference when published, else Barttorvik (2008+).
    for stat in ("bpm", "obpm", "dbpm"):
        f[stat] = last[stat].fillna(last[f"bart_{stat}"])
    f["team_srs"] = last["team_srs"]
    f["team_sos"] = last["team_sos"]
    f = f.join(_career_college(college))
    f["class_year"] = last["class"].map(CLASS_YEARS)
    f["last_college_season"] = last["season"]

    # Shot types (Barttorvik play-by-play, 2010+): share of FGA at the rim / midrange.
    fga = last["fga"].where(last["fga"] > 0)
    f["rim_rate"] = last["bart_rim_att"] / fga
    f["mid_rate"] = last["bart_mid_att"] / fga
    f["rim_fg_pct"] = last["bart_rim_made"] / last["bart_rim_att"].where(last["bart_rim_att"] > 0)
    f["dunk_rate"] = last["bart_dunks_att"] / fga

    # Position bucket: college listing, else combine, else BBRef listing.
    pos = last["pos"].fillna(cb["combine_position"]).fillna(players["pos"].str.split("-").str[0])
    f["position"] = pos.map(POSITION_BUCKET).fillna("unknown")
    for b in ("guard", "forward", "big"):
        f[f"pos_{b}"] = (f["position"] == b).astype(int)
    for src in ("college", "high_school", "other_team"):
        f[f"src_{src}"] = (f["prospect_source"] == src).astype(int)
    return f.reset_index()


def run(s: Settings) -> pd.DataFrame:
    f = build(s)
    write_table(f, "modeled", "features", "predraft", s)
    drafted = f[f["drafted"]]
    miss = drafted[STATS_FEATURES + SHOT_TYPE].isna().mean().sort_values()
    log.info(
        "pre-draft features: %d players; most-missing: %s", len(f), miss.tail(5).round(2).to_dict()
    )
    return f
