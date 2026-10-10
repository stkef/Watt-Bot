"""Polite one-time PDF download (FR-1).

One request at a time, a delay between requests, a named User-Agent, no crawling: only the
pinned URL of each document in metadata.csv. A document already on disk is never fetched again,
so a second run makes zero requests. Failures are logged by id.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import Config


def pdf_path(cfg: Config, doc_id: str) -> Path:
    return cfg.pdf_dir / f"{doc_id}.pdf"


def fetch_all(meta: pd.DataFrame, cfg: Config, delay: float = 3.0, timeout: float = 60.0,
              only: set[str] | None = None, sleep=time.sleep) -> dict:
    cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
    report = {"downloaded": [], "cached": [], "failed": {}}
    first = True
    for row in meta.itertuples():
        if only and row.id not in only:
            continue
        dest = pdf_path(cfg, row.id)
        if dest.exists() and dest.stat().st_size > 1000:
            report["cached"].append(row.id)
            continue
        if not first:
            sleep(delay)
        first = False
        try:
            req = urllib.request.Request(row.url, headers={"User-Agent": cfg.user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            if not body.startswith(b"%PDF"):
                raise ValueError("response is not a PDF (missing %PDF header)")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(dest)
            report["downloaded"].append(row.id)
        except Exception as e:  # noqa: BLE001 - log by id and carry on
            report["failed"][row.id] = f"{type(e).__name__}: {e}"
    (cfg.cache_dir / "fetch_report.json").write_text(json.dumps(report, indent=2))
    return report
