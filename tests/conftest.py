from pathlib import Path

import pytest

from draft_dna.config import Settings, load_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose data paths point into a temporary directory."""
    return load_settings(root=tmp_path)
