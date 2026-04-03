"""Config dataclass for HyperAtlas v4 pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    start_year: int = 2000
    end_year: int = 2024
    outdir: Path = field(default_factory=lambda: Path("v4_output"))
    bundle_dir: Path = field(default_factory=lambda: Path("bundle"))
    max_countries: int | None = None
    session_sleep_sec: float = 0.15
    seed: int = 2026
    n_iter: int = 3000
    burn: int = 1000
    thin: int = 2
    user_agent: str = "HyperAtlas/4.0 (research)"
    max_response_bytes: int = 50_000_000
