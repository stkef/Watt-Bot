"""Run the official Score.py on predictions and list the wrong ids (FR-12)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from .data import read_csv


def load_scorer(data_dir: Path):
    spec = importlib.util.spec_from_file_location("wattbot_official_score", Path(data_dir) / "Score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_predictions(solution: pd.DataFrame, pred: pd.DataFrame, data_dir: Path, verbose: bool = True) -> dict:
    S = load_scorer(data_dir)
    sol, sub = solution.copy(), pred.copy()
    overall = S.score(sol.copy(), sub.copy(), "id", verbose=verbose)
    sol_i = solution.astype(str).set_index("id")
    sub_i = pred.astype(str).set_index("id")
    wrong_value, wrong_ref = [], []
    for rid in sol_i.index.intersection(sub_i.index):
        bits = S.row_bits(sol_i.loc[rid], sub_i.loc[rid])
        if not bits["val"]:
            wrong_value.append(rid)
        if bits["ref"] < 1.0:
            wrong_ref.append(rid)
    if verbose:
        print(f"wrong answer_value ({len(wrong_value)}): {' '.join(wrong_value)}")
        print(f"imperfect ref_id   ({len(wrong_ref)}): {' '.join(wrong_ref)}")
    return {"overall": overall, "breakdown": dict(S.LAST_BREAKDOWN),
            "wrong_value": wrong_value, "wrong_ref": wrong_ref}


def score_files(solution_csv: Path, pred_csv: Path, data_dir: Path, verbose: bool = True) -> dict:
    return score_predictions(read_csv(solution_csv), read_csv(pred_csv), data_dir, verbose)
