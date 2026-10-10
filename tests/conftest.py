import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
has_data = pytest.mark.skipif(not (DATA / "train_QA.csv").exists(), reason="competition data not present")


class FakeLLM:
    """Deterministic stand-in for Gemini: returns a scripted JSON per question substring."""

    def __init__(self, script: dict[str, dict], default: dict | None = None):
        self.script, self.default, self.prompts = script, default, []

    def generate(self, system, prompt):
        self.prompts.append(prompt)
        q = prompt.rsplit("Question:", 1)[-1]
        for key, resp in self.script.items():
            if key in q:
                return json.dumps(resp)
        return json.dumps(self.default or {"answerable": False, "explanation": "not in excerpts"})


def hash_embed(texts):
    """Bag-of-words hashing embedder: deterministic, offline."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash(w) % 256] += 1.0
    return out


@pytest.fixture
def fake_llm():
    return FakeLLM
