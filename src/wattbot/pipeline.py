"""End-to-end: questions -> retrieval -> LLM -> rows, with a full per-question trace (Logged NFR)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .answer import build_prompt, build_row, parse_response, refusal_row, SYSTEM
from .chunk import Chunk, chunk_document
from .config import Config, SUBMISSION_COLUMNS
from .fewshot import FewShotPool
from .parse import load_pages
from .retrieve import HybridIndex, EmbedFn


# ───────────────────────── index persistence ─────────────────────────
def build_chunks(doc_ids: list[str], cfg: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in doc_ids:
        chunks.extend(chunk_document(d, load_pages(cfg, d)))
    return chunks


def save_index(idx: HybridIndex, cfg: Config) -> None:
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in idx.chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    if idx.emb is not None:
        np.save(cfg.index_dir / "emb.npy", idx.emb)


def load_index(cfg: Config, embed_fn: EmbedFn | None) -> HybridIndex:
    lines = (cfg.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [Chunk(**json.loads(l)) for l in lines if l.strip()]
    emb_path = cfg.index_dir / "emb.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    return HybridIndex(chunks, emb, embed_fn if emb is not None else None)


# ───────────────────────────── answering ─────────────────────────────
def answer_one(q: dict, index: HybridIndex, llm, pool: FewShotPool | None,
               url_by_id: dict[str, str], k_ctx: int = 12, k_shots: int = 3) -> tuple[dict, dict]:
    trace: dict = {"id": q["id"], "question": q["question"]}
    try:
        retrieved = index.search(q["question"], top_k=k_ctx)
        shots = pool.select(q["id"], q["question"], k=k_shots) if pool else []
        prompt, by_label = build_prompt(q["question"], q.get("answer_unit", ""), retrieved, shots)
        trace.update(retrieved=[c.chunk_id for c in retrieved], shots=[s["id"] for s in shots], prompt=prompt)
        raw = llm.generate(SYSTEM, prompt)
        trace["raw_response"] = raw
        row = build_row(q, parse_response(raw), by_label, retrieved, url_by_id)
    except Exception as e:  # noqa: BLE001 - a failed question becomes a refusal, never a crash
        trace["error"] = f"{type(e).__name__}: {e}"
        row = refusal_row(q, f"Processing failed ({type(e).__name__}); treated as unanswerable.")
    trace["row"] = row
    return row, trace


def run_questions(df: pd.DataFrame, index: HybridIndex, llm, pool: FewShotPool | None,
                  url_by_id: dict[str, str], log_dir: Path | None = None, workers: int = 4,
                  limit: int | None = None) -> pd.DataFrame:
    qs = df.to_dict("records")[:limit]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def work(q):
        row, trace = answer_one(q, index, llm, pool, url_by_id)
        if log_dir:
            (Path(log_dir) / f"{q['id']}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1))
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, qs))
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def write_submission(df: pd.DataFrame, path: Path) -> None:
    """Exact test_Q.csv column order; every explanation non-empty (Kaggle validation)."""
    out = df[SUBMISSION_COLUMNS].copy()
    if out["explanation"].str.strip().eq("").any():
        raise ValueError("a row has an empty explanation; Kaggle would reject the file")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
