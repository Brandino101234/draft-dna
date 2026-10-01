"""Typed access to config/settings.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "settings.yaml"


class Paths(BaseModel):
    raw: Path
    staging: Path
    modeled: Path
    database: Path
    reports: Path

    def resolved(self, root: Path) -> Paths:
        return Paths(**{k: root / v for k, v in self.model_dump().items()})


class DraftClasses(BaseModel):
    training: tuple[int, int]
    in_progress: tuple[int, int]
    live: list[int]

    @model_validator(mode="after")
    def _ordered_and_disjoint(self) -> DraftClasses:
        t0, t1 = self.training
        p0, p1 = self.in_progress
        if not (t0 <= t1 < p0 <= p1):
            raise ValueError("training and in_progress ranges must be ordered and disjoint")
        if any(y <= p1 for y in self.live):
            raise ValueError("live draft classes must come after in_progress classes")
        return self

    def training_years(self) -> list[int]:
        return list(range(self.training[0], self.training[1] + 1))

    def in_progress_years(self) -> list[int]:
        return list(range(self.in_progress[0], self.in_progress[1] + 1))

    def all_years(self) -> list[int]:
        return self.training_years() + self.in_progress_years() + list(self.live)


class RateLimit(BaseModel):
    min_seconds_between_requests: float


class HttpSettings(BaseModel):
    timeout_seconds: float
    max_retries: int
    backoff_seconds: float


class LoggingSettings(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    paths: Paths
    draft_classes: DraftClasses
    current_nba_season: int
    rate_limits: dict[str, RateLimit]
    http: HttpSettings
    logging: LoggingSettings


def load_settings(path: Path = DEFAULT_CONFIG, root: Path = REPO_ROOT) -> Settings:
    with path.open() as f:
        raw = yaml.safe_load(f)
    settings = Settings.model_validate(raw)
    settings.paths = settings.paths.resolved(root)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
