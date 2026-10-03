"""Command-line entry point: `draft-dna <command>`."""

from __future__ import annotations

from typing import Annotated

import typer

from draft_dna.config import get_settings
from draft_dna.ingest.run import STEPS
from draft_dna.logging_utils import get_logger, setup_logging

app = typer.Typer(no_args_is_help=True, help="Draft DNA data and modeling pipeline.")
log = get_logger(__name__)


@app.callback()
def main() -> None:
    setup_logging(get_settings().logging.level)


@app.command()
def info() -> None:
    """Print resolved configuration."""
    s = get_settings()
    log.info("database: %s", s.paths.database)
    log.info(
        "draft classes: training %s, in-progress %s, live %s",
        s.draft_classes.training,
        s.draft_classes.in_progress,
        s.draft_classes.live,
    )


@app.command()
def ingest(
    steps: Annotated[
        list[str] | None, typer.Argument(help=f"Steps to run (default: all): {list(STEPS)}")
    ] = None,
) -> None:
    """Pull raw data from sources. Resumable: cached pages are never re-fetched."""
    for name in steps or list(STEPS):
        if name not in STEPS:
            raise typer.BadParameter(f"unknown step {name!r}; choose from {list(STEPS)}")
        log.info("== ingest %s", name)
        STEPS[name]()


@app.command()
def transform() -> None:
    """Rebuild staging, crosswalk, modeled tables and DuckDB from raw (no network)."""
    from draft_dna.build import transform as run_transform

    run_transform()


@app.command()
def report() -> None:
    """Write reports for phases 1-6 and the recruiting-rank analysis to reports/."""
    from draft_dna.eval import (
        coverage,
        outcomes_report,
        phase3_report,
        phase4_report,
        phase5_report,
        phase6_report,
        recruits,
    )

    coverage.run()
    outcomes_report.run()
    phase3_report.run()
    phase4_report.run()
    phase5_report.run()
    phase6_report.run()
    recruits.run(get_settings())


@app.command()
def backtest(refresh: bool = False) -> None:
    """Phase 3: run every model through the rolling backtest (cached unless --refresh)."""
    from draft_dna.eval import phase3

    s = get_settings()
    phase3.predictions(s, phase3.load_frame(s), refresh=refresh)


@app.command()
def project() -> None:
    """Phase 3: build per-player projections and comps (as of each draft night)."""
    from draft_dna import db
    from draft_dna.features import predraft
    from draft_dna.models import projections

    s = get_settings()
    predraft.run(s)
    projections.run(s)
    db.load(s)


@app.command()
def grade() -> None:
    """Phase 7: grade every player and build trajectory bands, plays-like comps, style map."""
    from draft_dna import db
    from draft_dna.grading import extras, tracker
    from draft_dna.grading import run as grading
    from draft_dna.ingest.storage import write_table

    s = get_settings()
    grading.run(s)
    extras.run(s)
    write_table(tracker.rookie_tracker(s), "modeled", "grading", "rookie_tracker", s)
    from draft_dna.eval import accuracy, rim_confirmation
    from draft_dna.grading import player_metrics

    accuracy.run(s)
    player_metrics.run(s)
    rim_confirmation.run(s)
    from draft_dna.grading import pick_trades

    pick_trades.run(s)
    from draft_dna.grading import drafting

    drafting.run(s)  # writes status; runs the D029 test once a class is ready
    db.load(s)


SAMPLE_CARDS = ["dybanaj01", "peterda02", "boozeca02", "flaggco01", "wembavi01", "banchpa01"]


@app.command()
def cards(all_recent: bool = True) -> None:
    """Phase 7: render prospect cards (2022-2026 classes) and refresh the sample cards."""
    from draft_dna.ingest.storage import read_table
    from draft_dna.viz import cards as card_viz

    s = get_settings()
    grades = read_table("modeled", "grading", "grades", s)
    ids = grades.loc[grades["draft_year"] >= 2022, "bbref_id"].tolist() if all_recent else []
    out = card_viz.render_many(s, ids, s.paths.modeled / "cards")
    card_viz.render_many(s, SAMPLE_CARDS, s.paths.reports / "cards")
    log.info(
        "rendered %d cards to %s (+ %d samples in reports/cards)",
        len(out),
        s.paths.modeled / "cards",
        len(SAMPLE_CARDS),
    )


@app.command()
def bundle() -> None:
    """Export the small tables the deployed app reads into app/bundle/."""
    from draft_dna import app_bundle

    app_bundle.export(get_settings())


@app.command()
def build() -> None:
    """Ingest every source (cached pages are reused) and run all transforms."""
    from draft_dna.build import build_all

    build_all()


@app.command()
def refresh(skip_ingest: bool = False) -> None:
    """In-season update: re-pull current-season pages, rebuild tables, regrade, redraw cards.

    Only pages for the season in progress are re-fetched (older pages are cached forever),
    so this makes a handful of requests. Draft-night projections never change. Ends by
    re-exporting app/bundle; commit and push it to update the public app.
    """
    from draft_dna.build import transform as run_transform

    if not skip_ingest:
        log.info("== ingest bbref-league (current season only re-fetched)")
        STEPS["bbref-league"]()
    from draft_dna import app_bundle
    from draft_dna.ingest.storage import read_table

    run_transform()
    grade()
    cards()
    s = get_settings()
    app_bundle.export(s)
    t = read_table("modeled", "grading", "rookie_tracker", s)
    counts = t["status"].value_counts().to_dict()
    log.info("%d class tracker: %s", s.draft_classes.live[-1], counts)
    movers = t[t["status"].str.startswith("pacing")].head(10)
    for _, r in movers.iterrows():
        log.info(
            "  #%d %s: pace %.2f vs range %.2f-%.2f (%s)",
            int(r["pick"]),
            r["player_name"],
            r["value_pace"],
            r["projected_floor"],
            r["projected_ceiling"],
            r["status"],
        )


if __name__ == "__main__":
    app()
