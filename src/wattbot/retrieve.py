"""Hybrid retrieval (FR-5): BM25 for exact strings + dense embeddings, fused with RRF,
at most `per_doc` chunks per document."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Sequence

import numpy as np

from .chunk import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case tokens that keep 'llama-3.1-405b' / 'tco2e' / 'h100' whole, plus their parts,
    so both the exact string and its pieces match."""
    out: list[str] = []
    for t in _TOKEN.findall(text.lower()):
        out.append(t)
        if re.search(r"[.\-_/]", t):
            out.extend(p for p in re.split(r"[.\-_/]", t) if p)
    return out


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(self.tfs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.inv[t].append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                tf = self.tfs[i][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1.0))
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


EmbedFn = Callable[[list[str]], np.ndarray]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None,
                 embed_fn: EmbedFn | None = None):
        self.chunks = chunks
        self.bm25 = BM25([c.text for c in chunks])
        self.embed_fn = embed_fn
        if embeddings is not None:
            e = np.asarray(embeddings, dtype=np.float32)
            self.emb = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-9, None)
        else:
            self.emb = None

    @classmethod
    def build(cls, chunks: list[Chunk], embed_fn: EmbedFn | None, batch: int = 64) -> "HybridIndex":
        emb = None
        if embed_fn is not None and chunks:
            parts = [embed_fn([c.text for c in chunks[i:i + batch]]) for i in range(0, len(chunks), batch)]
            emb = np.vstack(parts)
        return cls(chunks, emb, embed_fn)

    def search(self, query: str, top_k: int = 12, per_doc: int = 4, pool: int = 60) -> list[Chunk]:
        if not self.chunks:
            return []
        bm = self.bm25.scores(query)
        rankings = [sorted(bm, key=lambda i: -bm[i])[:pool]]
        if self.emb is not None and self.embed_fn is not None:
            q = np.asarray(self.embed_fn([query]), dtype=np.float32)[0]
            q = q / max(float(np.linalg.norm(q)), 1e-9)
            sims = self.emb @ q
            rankings.append(list(np.argsort(-sims)[:pool]))
        picked, per = [], Counter()
        for idx, _ in rrf(rankings):
            ch = self.chunks[idx]
            if per[ch.doc_id] >= per_doc:
                continue
            per[ch.doc_id] += 1
            picked.append(ch)
            if len(picked) >= top_k:
                break
        return picked

    def doc_recall(self, query: str, gold_docs: set[str], top_k: int = 12) -> float:
        """Share of gold documents present in the top_k chunks (retrieval-only check, no LLM)."""
        if not gold_docs:
            return 1.0
        got = {c.doc_id for c in self.search(query, top_k)}
        return len(got & gold_docs) / len(gold_docs)
