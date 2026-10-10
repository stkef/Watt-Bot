"""Page-bounded chunking (FR-4).

* a chunk never crosses a page, so a quote always maps to one (document, page);
* prose is windowed (~260 words, 50 overlap);
* a markdown table is kept whole; an oversized table is split by rows with its header repeated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

CHUNK_WORDS = 260
OVERLAP_WORDS = 50
MAX_TABLE_WORDS = 600


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    kind: str = "text"  # text | table

    def to_dict(self) -> dict:
        return asdict(self)


def _blocks(text: str) -> list[tuple[str, str]]:
    """Split page text into ('table'|'text', block) pieces, keeping tables contiguous."""
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_kind = "text"

    def flush():
        nonlocal cur
        if cur:
            out.append((cur_kind, "\n".join(cur).strip()))
            cur = []

    for line in text.splitlines():
        kind = "table" if line.lstrip().startswith("|") else "text"
        if kind != cur_kind:
            flush()
            cur_kind = kind
        cur.append(line)
    flush()
    return [(k, b) for k, b in out if b]


def _split_table(block: str) -> list[str]:
    if len(block.split()) <= MAX_TABLE_WORDS:
        return [block]
    lines = block.splitlines()
    header = lines[:2] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ") else lines[:1]
    body = lines[len(header):]
    parts, cur, words = [], [], 0
    for ln in body:
        w = len(ln.split())
        if cur and words + w > MAX_TABLE_WORDS:
            parts.append("\n".join(header + cur))
            cur, words = [], 0
        cur.append(ln)
        words += w
    if cur:
        parts.append("\n".join(header + cur))
    return parts


def chunk_page(doc_id: str, page: int, text: str,
               words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(t: str, kind: str):
        chunks.append(Chunk(f"{doc_id}:p{page}:c{len(chunks)}", doc_id, page, t, kind))

    prose: list[str] = []

    def flush_prose():
        if not prose:
            return
        toks = " ".join(prose).split()
        prose.clear()
        step = max(words - overlap, 1)
        start = 0
        while start < len(toks):
            add(" ".join(toks[start:start + words]), "text")
            if start + words >= len(toks):
                break
            start += step

    for kind, block in _blocks(text):
        if kind == "table":
            flush_prose()
            for part in _split_table(block):
                add(part, "table")
        else:
            prose.append(block)
    flush_prose()
    return chunks


def chunk_document(doc_id: str, pages: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        out.extend(chunk_page(doc_id, int(p["page"]), p.get("text", "")))
    return out
