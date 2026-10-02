"""Command-line entry point: `draft-dna <command>`.

Commands for later phases are stubs that exit with a clear message.
"""

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
    """Write reports for phases 1-5 to reports/."""
    from draft_dna.eval import (
        coverage,
        outcomes_report,
        phase3_report,
        phase4_report,
        phase5_report,
    )

    coverage.run()
    outcomes_report.run()
    phase3_report.run()
    phase4_report.run()
    phase5_report.run()


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
def build() -> None:
    """Ingest every source (cached pages are reused) and run all transforms."""
    from draft_dna.build import build_all

    build_all()


@app.command()
def refresh() -> None:
    """Pull new games and regrade players during the season. Phase 7."""
    raise typer.Exit(_not_yet("refresh", 7))


def _not_yet(command: str, phase: int) -> int:
    log.error("`%s` is implemented in Phase %d.", command, phase)
    return 1


if __name__ == "__main__":
    app()
