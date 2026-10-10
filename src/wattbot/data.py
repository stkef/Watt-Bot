"""Loading the competition CSVs and the list-valued ref columns."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from .config import BLANK


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read as strings and keep 'NA'/'' literal (pandas would turn them into NaN)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_train(data_dir: Path) -> pd.DataFrame:
    return read_csv(Path(data_dir) / "train_QA.csv")


def load_test(data_dir: Path) -> pd.DataFrame:
    return read_csv(Path(data_dir) / "test_Q.csv")


def load_metadata(data_dir: Path) -> pd.DataFrame:
    return read_csv(Path(data_dir) / "metadata.csv")


def parse_ref_list(s: str) -> list[str]:
    """"['a', 'b']" | "a" | "is_blank" | "" -> list of ids (blank -> [])."""
    s = (s or "").strip()
    if not s or s.lower() == BLANK:
        return []
    if s.startswith("["):
        try:
            return [str(x).strip() for x in ast.literal_eval(s) if str(x).strip()]
        except (ValueError, SyntaxError):
            inner = s.strip("[]")
            return [t.strip().strip("'\"") for t in inner.split(",") if t.strip()]
    return [s]


def format_ref_list(items: list[str]) -> str:
    """Same shape as train_QA.csv: "['a', 'b']"; blank when empty."""
    return str(list(items)) if items else BLANK
