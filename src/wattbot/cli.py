"""Command line: fetch -> parse -> index -> run -> score (and recall for retrieval-only checks)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import Config


def _ensure_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    try:  # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient  # type: ignore
        os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception:  # noqa: BLE001
        pass


def _embedder(cfg: Config):
    from .llm import DiskCache, GeminiEmbedder
    return GeminiEmbedder(cfg.embed_model, DiskCache(cfg.llm_cache_dir / "emb"))


def cmd_fetch(a, cfg):
    from .data import load_metadata
    from .fetch import fetch_all
    rep = fetch_all(load_metadata(cfg.data_dir), cfg, delay=a.delay)
    print(f"downloaded {len(rep['downloaded'])}, cached {len(rep['cached'])}, failed {len(rep['failed'])}")
    for k, v in rep["failed"].items():
        print(f"  FAILED {k}: {v}")


def cmd_parse(a, cfg):
    from .data import load_metadata
    from .parse import parse_all
    rep = parse_all(load_metadata(cfg.data_dir)["id"].tolist(), cfg)
    print(f"parsed {len(rep['parsed'])}, already parsed {len(rep['skipped'])}, failed {len(rep['failed'])}")
    for k, v in rep["failed"].items():
        print(f"  FAILED {k}: {v}")


def cmd_index(a, cfg):
    from .data import load_metadata
    from .pipeline import build_chunks, save_index
    from .retrieve import HybridIndex
    ids = [i for i in load_metadata(cfg.data_dir)["id"] if (cfg.parsed_dir / f"{i}.jsonl").exists()]
    chunks = build_chunks(ids, cfg)
    emb = None
    if not a.no_dense:
        _ensure_key()
        emb = _embedder(cfg)
    idx = HybridIndex.build(chunks, emb)
    save_index(idx, cfg)
    print(f"indexed {len(chunks)} chunks from {len(ids)} documents (dense={'no' if a.no_dense else 'yes'})")


def _load_idx(cfg, no_dense):
    from .pipeline import load_index
    if not no_dense:
        _ensure_key()
    return load_index(cfg, None if no_dense else _embedder(cfg))


def cmd_run(a, cfg):
    from .data import load_metadata, load_test, load_train
    from .fewshot import FewShotPool
    from .llm import DiskCache, GeminiLLM
    from .pipeline import run_questions, write_submission
    _ensure_key()
    meta = load_metadata(cfg.data_dir)
    urls = dict(zip(meta["id"], meta["url"]))
    train = load_train(cfg.data_dir)
    df = train if a.split == "train" else load_test(cfg.data_dir)
    llm = GeminiLLM(cfg.gen_model, DiskCache(cfg.llm_cache_dir / "gen"))
    out = run_questions(df, _load_idx(cfg, a.no_dense), llm, FewShotPool(train), urls,
                        log_dir=cfg.runs_dir / a.name / "logs", workers=a.workers, limit=a.limit)
    dest = cfg.runs_dir / a.name / f"{a.split}_predictions.csv"
    write_submission(out, dest)
    print(f"wrote {dest} ({len(out)} rows, {llm.calls} uncached model calls)")


def cmd_score(a, cfg):
    from .data import load_train
    from .score_local import score_predictions
    from .data import read_csv
    pred = read_csv(a.pred)
    sol = load_train(cfg.data_dir)
    sol = sol[sol["id"].isin(pred["id"])]  # score only the questions that were run
    score_predictions(sol, pred, cfg.data_dir)


def cmd_recall(a, cfg):
    from .data import load_train, parse_ref_list
    idx = _load_idx(cfg, a.no_dense)
    tr = load_train(cfg.data_dir)
    tr = tr[tr["is_NA"] != "1"]
    rec = [idx.doc_recall(r.question, set(parse_ref_list(r.ref_id)), top_k=a.k) for r in tr.itertuples()]
    print(f"recall@{a.k} of gold documents over {len(rec)} answerable train questions: {sum(rec)/len(rec):.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="wattbot")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch"); s.add_argument("--delay", type=float, default=3.0); s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("parse"); s.set_defaults(fn=cmd_parse)
    s = sub.add_parser("index"); s.add_argument("--no-dense", action="store_true"); s.set_defaults(fn=cmd_index)
    s = sub.add_parser("run")
    s.add_argument("--split", choices=["train", "test"], default="train")
    s.add_argument("--name", default="baseline"); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int); s.add_argument("--no-dense", action="store_true"); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("score"); s.add_argument("pred", type=Path); s.set_defaults(fn=cmd_score)
    s = sub.add_parser("recall"); s.add_argument("-k", type=int, default=12)
    s.add_argument("--no-dense", action="store_true"); s.set_defaults(fn=cmd_recall)
    a = p.parse_args(argv)
    a.fn(a, Config())


if __name__ == "__main__":
    sys.exit(main())
