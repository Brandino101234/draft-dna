"""End-to-end build: raw -> staging -> crosswalk -> modeled -> DuckDB."""

from __future__ import annotations

from draft_dna import db
from draft_dna.config import Settings, get_settings
from draft_dna.crosswalk import build as crosswalk
from draft_dna.features import era, predraft
from draft_dna.ingest.run import STEPS
from draft_dna.logging_utils import get_logger
from draft_dna.modeled import integrate
from draft_dna.outcomes import run as outcomes
from draft_dna.staging import bbref as stage_bbref
from draft_dna.staging import other as stage_other

log = get_logger(__name__)


def transform(s: Settings | None = None) -> None:
    """Everything after ingestion. Fast; safe to rerun any time."""
    s = s or get_settings()
    log.info("== staging")
    stage_bbref.run(s)
    stage_other.run(s)
    log.info("== crosswalk")
    crosswalk.run(s)
    log.info("== modeled")
    era.run(s)
    integrate.run(s)
    log.info("== outcomes")
    outcomes.run(s)
    predraft.run(s)
    log.info("== duckdb")
    db.load(s)


def build_all(s: Settings | None = None) -> None:
    s = s or get_settings()
    for name, step in STEPS.items():
        log.info("== ingest %s", name)
        step(s)
    transform(s)
