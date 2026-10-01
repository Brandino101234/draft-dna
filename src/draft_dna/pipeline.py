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
