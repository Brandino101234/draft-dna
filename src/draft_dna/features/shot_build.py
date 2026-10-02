"""Build Shot DNA tables from downloaded ESPN play-by-play.

Outputs (modeled layer):
- shots.espn_player_map:   ESPN athlete id -> bbref_id per college season
- shots.college_shots:     every linked pre-draft field-goal attempt, standardized
- features.shot_coverage:  per player: PBP attempts, attempts with coordinates, coverage
- features.shot_zones:     5-zone frequencies (coordinate-eligible players only)
- features.shot_maps:      flattened density maps (coordinate-eligible players only)

Eligibility for spatial features (D025): >= 100 field-goal attempts with valid
coordinates AND >= 40% of the player's play-by-play attempts carrying coordinates.
Everyone else falls back to tier-A shot-type features or stats only.
"""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings
from draft_dna.crosswalk import espn_players
from draft_dna.eval.shot_audit import SEASONS, classify
from draft_dna.features import shot_xy
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

ZONES = ["rim", "short_mid", "long_mid", "corner_three", "above_break_three"]


def load_espn(s: Settings, kind: str) -> pd.DataFrame:
    paths = [table_path("raw", "espn", f"{kind}_{y}", s) for y in SEASONS]
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(s: Settings) -> dict[str, pd.DataFrame]:
    shots = load_espn(s, "shots")
    rosters = load_espn(s, "rosters")
    if shots.empty:
        raise RuntimeError("no ESPN shots downloaded yet; run `draft-dna ingest espn-shots`")
    college = read_table("modeled", "core", "college_player_seasons", s)
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    college = college[college["bbref_id"].map(players["drafted"]).fillna(False).astype(bool)]
    team_map = read_table("raw", "espn", "team_seasons", s)
    pmap = espn_players.link(college, team_map, rosters, players["player_name"])

    linked = shots.merge(
        pmap[["season", "athlete_id", "bbref_id", "is_pre_draft"]], on=["season", "athlete_id"]
    )
    linked = linked[linked["is_pre_draft"]]
    linked["kind"] = [
        classify(t, x or "", v)
        for t, x, v in zip(linked["shot_type"], linked["text"], linked["score_value"], strict=True)
    ]
    std = shot_xy.standardize_espn(linked)
    std["zone"] = shot_xy.zone(std)

    cov = std.groupby("bbref_id").agg(
        pbp_fga=("game_id", "size"),
        xy_fga=("xy_valid", "sum"),
        first_season=("season", "min"),
        last_season=("season", "max"),
    )
    cov["xy_coverage"] = cov["xy_fga"] / cov["pbp_fga"]
    cov["xy_eligible"] = (cov["xy_fga"] >= shot_xy.MIN_XY_FGA) & (
        cov["xy_coverage"] >= shot_xy.MIN_XY_COVERAGE
    )
    eligible = std[std["bbref_id"].isin(cov.index[cov["xy_eligible"]]) & std["xy_valid"]]
    zones = eligible.groupby("bbref_id")["zone"].value_counts(normalize=True).unstack(fill_value=0)
    zones = zones.reindex(columns=ZONES, fill_value=0).add_prefix("zone_")
    maps = shot_xy.player_maps(eligible)
    return {
        "pmap": pmap,
        "shots": std,
        "coverage": cov.reset_index(),
        "zones": zones.reset_index(),
        "maps": maps.reset_index(names="bbref_id"),
    }


def run(s: Settings) -> dict[str, pd.DataFrame]:
    out = build(s)
    write_table(out["pmap"], "modeled", "shots", "espn_player_map", s)
    keep = [
        "bbref_id",
        "season",
        "game_id",
        "kind",
        "made",
        "assist_athlete_id",
        "is_three",
        "dx",
        "dy",
        "dist_ft",
        "zone",
        "xy_valid",
    ]
    write_table(out["shots"][keep], "modeled", "shots", "college_shots", s)
    write_table(out["coverage"], "modeled", "features", "shot_coverage", s)
    write_table(out["zones"], "modeled", "features", "shot_zones", s)
    write_table(out["maps"], "modeled", "features", "shot_maps", s)
    cov = out["coverage"]
    log.info(
        "shot DNA: %d players with PBP shots, %d coordinate-eligible",
        len(cov),
        int(cov["xy_eligible"].sum()),
    )
    return out
