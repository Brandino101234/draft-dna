from pathlib import Path

import pytest
import yaml

from draft_dna.config import DEFAULT_CONFIG, REPO_ROOT, DraftClasses, load_settings


def test_settings_load_and_resolve_paths() -> None:
    s = load_settings()
    assert s.paths.database.is_absolute()
    assert s.paths.raw == REPO_ROOT / "data" / "raw"


def test_draft_class_splits_cover_1996_to_2026_without_overlap() -> None:
    dc = load_settings().draft_classes
    years = dc.all_years()
    assert years == list(range(1996, 2027))
    assert len(set(years)) == len(years)


def test_overlapping_draft_classes_rejected() -> None:
    with pytest.raises(ValueError):
        DraftClasses(training=(1996, 2022), in_progress=(2022, 2025), live=[2026])


def test_rate_limits_respect_sports_reference_policy() -> None:
    # Sports-Reference bans clients above ~20 requests/minute.
    sr = load_settings().rate_limits["sports_reference"]
    assert 60 / sr.min_seconds_between_requests < 20


def test_data_and_secrets_are_gitignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()
    for pattern in [".env", "/data/raw/", "/data/staging/", "*.duckdb"]:
        assert pattern in ignored


def test_config_is_valid_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert "draft_classes" in raw
