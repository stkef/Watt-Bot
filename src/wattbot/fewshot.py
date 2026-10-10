"""Leave-one-out few-shot selection (FR-7).

Solved training questions teach the scorer's conventions (units, ranges, lists) better than
written rules alone. The one hard rule: a question must NEVER see its own answer, otherwise the
local score is inflated. `select_examples` enforces that in code by excluding the target by id
AND by identical question text (guards against duplicated rows under different ids).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOP = set("the a an of in on for to and or is are was were what which how many much does do did by with as at from that this it its be".split())


def _tok(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class FewShotPool:
    """TF-IDF cosine over solved questions."""

    def __init__(self, solved: pd.DataFrame):
        self.rows = solved.reset_index(drop=True)
        docs = [Counter(_tok(q)) for q in self.rows["question"]]
        df = Counter(t for d in docs for t in d)
        n = max(len(docs), 1)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def select(self, qid: str, question: str, k: int = 3) -> list[dict]:
        q = self._vec(Counter(_tok(question)))
        qnorm = question.strip().lower()
        scored = []
        for i, row in self.rows.iterrows():
            if row["id"] == qid or row["question"].strip().lower() == qnorm:
                continue  # leave-one-out: never show a question its own answer
            vec = self.vecs[i]
            sim = sum(w * vec.get(t, 0.0) for t, w in q.items())
            scored.append((sim, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for _, i in scored[:k]:
            r = self.rows.iloc[i]
            out.append({c: r[c] for c in ("id", "question", "answer", "answer_value", "answer_unit") if c in r})
        return out
