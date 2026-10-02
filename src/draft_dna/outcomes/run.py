"""Build all Phase 2 outcome tables (modeled.outcomes__*)."""

from __future__ import annotations

import json

import pandas as pd

from draft_dna.config import Settings
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger
from draft_dna.outcomes import survival, tiers, validate, value

log = get_logger(__name__)
P = value.PRIMARY
EARLY_NS = (3, 4)


def career_outcomes(
    otn: pd.DataFrame, players: pd.DataFrame, careers: pd.DataFrame
) -> pd.DataFrame:
    """One row per player: outcome at the latest observed N plus early-career outcomes."""
    latest = otn.sort_values("n").groupby("bbref_id").last()
    out = players.set_index("bbref_id")[["player_name", "drafted", "draft_year", "pick_overall",
                                         "draft_class_role"]].join(latest, how="left")  # fmt: skip
    out = out.rename(columns={"n": "n_observed"})
    out["n_observed"] = out["n_observed"].fillna(0).astype(int)
    for n in EARLY_NS:
        at = otn[otn["n"] == n].set_index("bbref_id")
        out[f"composite_y{n}"] = at[f"composite_{P}"]
        out[f"total_y{n}"] = at[f"total_{P}"]
        out[f"peak3_y{n}"] = at[f"peak3_{P}"]
    out["career_value"] = out[f"composite_{P}"]
    c = careers.set_index("bbref_id")[["duration", "ended", "seasons_played", "age_at_draft"]]
    out = out.join(c, how="left")
    return out.reset_index()


def run(s: Settings) -> dict[str, object]:
    players = read_table("modeled", "core", "players", s)
    seasons, otn, meta = value.build(s)
    training = players["draft_class_role"] == "training"

    # Tiers: calibrate on training classes at their latest N, then apply at every N.
    latest = otn[otn["bbref_id"].isin(players.loc[training, "bbref_id"])]
    latest = latest.sort_values("n").groupby("bbref_id").last()
    anchor = latest.apply(tiers.role_anchor, axis=1)
    cuts = tiers.calibrate(latest[f"peak3_{P}"], anchor)
    otn["tier_idx"] = tiers.assign(otn[f"peak3_{P}"], cuts)
    otn["tier"] = [tiers.TIERS[i] for i in otn["tier_idx"]]

    careers = survival.career_table(s)
    co = career_outcomes(otn, players, careers)
    co["tier"] = co["tier"].where(co["n_observed"] > 0)

    keep = ["bbref_id", "season", *[c for c in seasons.columns if c.startswith("value_")],
            "bpm_shrunk", "start_share"]  # fmt: skip
    write_table(seasons[keep], "modeled", "outcomes", "season_values", s)
    write_table(otn, "modeled", "outcomes", "outcomes_through_n", s)
    write_table(co, "modeled", "outcomes", "career_outcomes", s)
    write_table(careers, "modeled", "outcomes", "careers", s)

    agreement = float((anchor.to_numpy() == tiers.assign(latest[f"peak3_{P}"], cuts)).mean())
    params = {
        "primary_metric": P,
        "peak_weight": value.PEAK_WEIGHT,
        "tier_names": tiers.TIERS,
        "tier_cuts_peak3": cuts,
        "tier_anchor_agreement": round(agreement, 3),
        "factor_loadings": meta["factor_loadings"],
    }
    path = table_path("modeled", "outcomes", "params", s).with_suffix(".json")
    path.write_text(json.dumps(params, indent=2))
    log.info("tier cuts %s (anchor agreement %.3f)", cuts, agreement)
    return params


def validation_results(s: Settings, n_boot: int = validate.N_BOOT) -> pd.DataFrame:
    players = read_table("modeled", "core", "players", s)
    otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
    return validate.evaluate(validate.validation_frame(s, otn, players), n_boot=n_boot)
