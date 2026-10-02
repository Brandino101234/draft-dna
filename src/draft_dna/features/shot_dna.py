"""Shot DNA, tier A: how a prospect scores, from shot *types* (no coordinates needed).

Sources:
- Barttorvik play-by-play splits (2010+): rim / midrange / three attempts and makes, dunks.
- ESPN play-by-play (2008+, filled in once downloaded): assisted vs unassisted makes.

Features use the final pre-draft college season for shot mix (how he played most
recently) and all pre-draft seasons for efficiency (more attempts, steadier rates).

Efficiency by zone is shrunk with empirical Bayes: each zone's percentage is pulled
toward the Division I average for that season, more strongly when the player has few
attempts. The prior comes from all D-I players that season (thousands of players, all
pre-draft information), so there is no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

MIN_FGA_FOR_PRIOR = 50
ZONES = {
    "rim": ("bart_rim_made", "bart_rim_att"),
    "mid": ("bart_mid_made", "bart_mid_att"),
    "three": ("bart_three_pm", "bart_three_pa"),
}
POP_ZONES = {
    "rim": ("rim_made", "rim_att"),
    "mid": ("mid_made", "mid_att"),
    "three": ("three_pm", "three_pa"),
}
SHOT_FEATURES = [
    "rim_rate",
    "mid_rate",
    "three_rate",
    "three_rate_rel",
    "dunk_share",
    "rim_fg_eb",
    "mid_fg_eb",
    "three_fg_eb",
    "rim_fg_eb_rel",
    "mid_fg_eb_rel",
    "three_fg_eb_rel",
]
ASSIST_FEATURES = [
    "assisted_rim_share",
    "assisted_mid_share",
    "assisted_three_share",
    "unassisted_share",
]


@dataclass(frozen=True)
class BetaPrior:
    """Beta(alpha, beta) prior for a success rate: mean m, strength kappa = alpha + beta
    (think of kappa as "how many attempts' worth of evidence the average is worth")."""

    mean: float
    kappa: float

    def shrink(self, made: np.ndarray, att: np.ndarray) -> np.ndarray:
        return (made + self.mean * self.kappa) / (att + self.kappa)


def fit_beta_prior(
    made: np.ndarray, att: np.ndarray, min_att: int = MIN_FGA_FOR_PRIOR
) -> BetaPrior:
    """Method of moments: true-talent variance = observed variance minus binomial noise."""
    ok = att >= min_att
    made, att = made[ok].astype(float), att[ok].astype(float)
    m = made.sum() / att.sum()
    p = made / att
    # Attempt-weighted variance of observed rates = true-talent variance + binomial noise,
    # where the (attempt-weighted) noise term is m(1-m) / mean(attempts).
    observed_var = np.average((p - m) ** 2, weights=att)
    noise = m * (1 - m) * len(att) / att.sum()
    true_var = max(observed_var - noise, 1e-6)
    kappa = max(m * (1 - m) / true_var - 1, 1.0)
    return BetaPrior(float(m), float(kappa))


def season_priors(bart: pd.DataFrame) -> pd.DataFrame:
    """Per season and zone: D-I prior (mean, kappa) and D-I average shot mix."""
    rows = []
    for season, g in bart.groupby("season"):
        fga = g["rim_att"] + g["mid_att"] + g["three_pa"]
        if g["rim_att"].isna().all():
            continue
        row = {"season": season, "pop_three_rate": g["three_pa"].sum() / fga.sum()}
        for zone, (made, att) in POP_ZONES.items():
            prior = fit_beta_prior(g[made].fillna(0).to_numpy(), g[att].fillna(0).to_numpy())
            row[f"{zone}_mean"], row[f"{zone}_kappa"] = prior.mean, prior.kappa
        rows.append(row)
    return pd.DataFrame(rows).set_index("season")


def build(s: Settings) -> pd.DataFrame:
    college = read_table("modeled", "core", "college_player_seasons", s)
    bart = read_table("staging", "barttorvik", "player_seasons", s)
    priors = season_priors(bart)
    c = college[college["is_pre_draft"] & college["bart_rim_att"].notna()].copy()
    c = c.sort_values(["bbref_id", "season", "mp"])
    last = c.groupby("bbref_id").tail(1).set_index("bbref_id")
    career = c.groupby("bbref_id")[[m for z in ZONES.values() for m in z]].sum()

    f = pd.DataFrame(index=last.index)
    f["shot_season"] = last["season"]
    fga = last["bart_rim_att"] + last["bart_mid_att"] + last["bart_three_pa"]
    f["shot_fga"] = fga
    f["rim_rate"] = last["bart_rim_att"] / fga
    f["mid_rate"] = last["bart_mid_att"] / fga
    f["three_rate"] = last["bart_three_pa"] / fga
    f["three_rate_rel"] = f["three_rate"] - f["shot_season"].map(priors["pop_three_rate"])
    f["dunk_share"] = last["bart_dunks_att"] / last["bart_rim_att"].where(last["bart_rim_att"] > 0)
    for zone, (made, att) in ZONES.items():
        mean = f["shot_season"].map(priors[f"{zone}_mean"])
        kappa = f["shot_season"].map(priors[f"{zone}_kappa"])
        eb = (career[made] + mean * kappa) / (career[att] + kappa)
        f[f"{zone}_fg_eb"] = eb
        f[f"{zone}_fg_eb_rel"] = eb - mean  # above/below that season's D-I average
        f[f"{zone}_att_career"] = career[att]
    f = f.join(assisted_features(s), how="left")
    return f.reset_index()


def assisted_features(s: Settings) -> pd.DataFrame:
    """Share of each player's made field goals that were assisted, by shot type (ESPN).

    Empty until the ESPN download has produced `raw/espn/shots_<season>` tables.
    """
    from draft_dna.eval.shot_audit import SEASONS, classify

    tables = [table_path("raw", "espn", f"shots_{y}", s) for y in SEASONS]
    tables = [t for t in tables if t.exists()]
    if not tables or not table_path("modeled", "shots", "espn_player_map", s).exists():
        return pd.DataFrame(columns=ASSIST_FEATURES)
    shots = pd.concat([pd.read_parquet(t) for t in tables], ignore_index=True)
    pmap = read_table("modeled", "shots", "espn_player_map", s)
    shots = shots.merge(pmap, left_on=["season", "athlete_id"], right_on=["season", "athlete_id"])
    shots = shots[shots["is_pre_draft"]]
    shots["kind"] = [
        classify(t, x or "", v)
        for t, x, v in zip(shots["shot_type"], shots["text"], shots["score_value"], strict=True)
    ]
    made = shots[shots["made"]]
    made = made.assign(
        assisted=made["assist_athlete_id"].notna(),
        zone=made["kind"].map(
            {"layup": "rim", "dunk": "rim", "tip": "rim", "jumper": "mid", "three": "three"}
        ),
    )
    out = made.pivot_table(index="bbref_id", columns="zone", values="assisted", aggfunc="mean")
    out = out.rename(columns={z: f"assisted_{z}_share" for z in ("rim", "mid", "three")})
    out["unassisted_share"] = 1 - made.groupby("bbref_id")["assisted"].mean()
    return out.reindex(columns=ASSIST_FEATURES)


def run(s: Settings) -> pd.DataFrame:
    f = build(s)
    write_table(f, "modeled", "features", "shot_dna", s)
    log.info("shot DNA (tier A): %d players", len(f))
    return f
