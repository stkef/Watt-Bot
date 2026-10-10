"""Paths and model names. Everything is overridable by environment variable so the same
code runs locally and in a Kaggle notebook (where the competition data and the private
parsed-corpus dataset live under /kaggle/input)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BLANK = "is_blank"

# Columns written to the submission, in the order of test_Q.csv (the file we are asked to fill).
SUBMISSION_COLUMNS = [
    "id", "question", "answer", "answer_value", "answer_unit",
    "ref_id", "ref_url", "supporting_materials", "explanation", "Cohort",
]


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=lambda: _env_path("WATTBOT_DATA", "data"))
    cache_dir: Path = field(default_factory=lambda: _env_path("WATTBOT_CACHE", "cache"))
    runs_dir: Path = field(default_factory=lambda: _env_path("WATTBOT_RUNS", "runs"))
    gen_model: str = field(default_factory=lambda: os.environ.get("WATTBOT_GEN_MODEL", "gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: os.environ.get("WATTBOT_EMBED_MODEL", "gemini-embedding-001"))
    user_agent: str = "wattbot-research/0.1 (Kaggle WattBot 2026; polite single-threaded fetch)"

    @property
    def pdf_dir(self) -> Path:
        return self.cache_dir / "pdf"

    @property
    def parsed_dir(self) -> Path:
        return self.cache_dir / "parsed"

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def index_dir(self) -> Path:
        return self.cache_dir / "index"
