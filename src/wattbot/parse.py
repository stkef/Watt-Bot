"""PDF -> per-page text (FR-2). Text in reading order, tables as markdown, de-hyphenated.

Parsed pages are cached as one JSONL per document so a later run (or a Kaggle notebook that
cannot reach the internet) never touches the PDFs again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Config

_HYPHEN = re.compile(r"(\w)-\n(\w)")


def dehyphenate(text: str) -> str:
    return _HYPHEN.sub(r"\1\2", text)


def table_to_markdown(rows: list[list]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r and any(c for c in r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join("---" for _ in range(width)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def parse_pdf(path: str | Path) -> list[dict]:
    import pdfplumber  # lazy: only needed when actually parsing

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = []
            try:
                found = page.find_tables()
            except Exception:  # noqa: BLE001
                found = []
            bboxes = [t.bbox for t in found]
            md_tables = []
            for t in found:
                try:
                    md = table_to_markdown(t.extract())
                except Exception:  # noqa: BLE001
                    md = ""
                if md:
                    md_tables.append(md)
            src = page
            if bboxes:
                def outside(obj, bboxes=bboxes):
                    if "x0" not in obj or "top" not in obj:
                        return True
                    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
                    return not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in bboxes)
                src = page.filter(outside)
            text = dehyphenate(src.extract_text() or "")
            if md_tables:
                text = text.rstrip() + "\n\n" + "\n\n".join(md_tables)
            words = len(text.split())
            pages.append({
                "page": i,
                "text": text.strip(),
                # candidate for the vision pass (FR-3): drawn images/charts with little prose
                "figure_candidate": bool(page.images) and words < 250 or bool(re.search(r"\bFig(?:ure|\.)\s*\d", text) and page.images),
            })
    return pages


def parsed_path(cfg: Config, doc_id: str) -> Path:
    return cfg.parsed_dir / f"{doc_id}.jsonl"


def save_pages(cfg: Config, doc_id: str, pages: list[dict]) -> None:
    cfg.parsed_dir.mkdir(parents=True, exist_ok=True)
    with parsed_path(cfg, doc_id).open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_pages(cfg: Config, doc_id: str) -> list[dict]:
    p = parsed_path(cfg, doc_id)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_all(meta_ids: list[str], cfg: Config) -> dict:
    """Parse every cached PDF that has no parsed copy yet. Report page counts and failures by id."""
    report = {"parsed": {}, "skipped": [], "failed": {}}
    for doc_id in meta_ids:
        if parsed_path(cfg, doc_id).exists():
            report["skipped"].append(doc_id)
            continue
        pdf = cfg.pdf_dir / f"{doc_id}.pdf"
        if not pdf.exists():
            report["failed"][doc_id] = "pdf missing"
            continue
        try:
            pages = parse_pdf(pdf)
            if not any(p["text"] for p in pages):
                raise ValueError("no extractable text (scanned? needs OCR)")
            save_pages(cfg, doc_id, pages)
            report["parsed"][doc_id] = len(pages)
        except Exception as e:  # noqa: BLE001
            report["failed"][doc_id] = f"{type(e).__name__}: {e}"
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    (cfg.cache_dir / "parse_report.json").write_text(json.dumps(report, indent=2))
    return report
