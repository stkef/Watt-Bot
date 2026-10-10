"""LLM access: plain REST calls, no framework, every response cached on disk by prompt hash.

The API key is read from GEMINI_API_KEY (Kaggle Secrets exposes it via UserSecretsClient; the
CLI copies it into the environment). It is never logged or written to the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "")
    if not k:
        raise RuntimeError("GEMINI_API_KEY is not set (use Kaggle Secrets or export it locally)")
    return k


def _post(url: str, payload: dict, retries: int = 5, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "x-goog-api-key": _key()})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            raise RuntimeError(f"HTTP {e.code} from model API") from None  # never echo headers/key
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str):
        p = self._path(key)
        return json.loads(p.read_text())["v"] if p.exists() else None

    def put(self, key: str, value) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps({"v": value}))
        tmp.replace(self._path(key))


def prompt_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


class GeminiLLM:
    """temperature 0, JSON output, cached by hash of (model, system, prompt)."""

    def __init__(self, model: str, cache: DiskCache, temperature: float = 0.0):
        self.model, self.cache, self.temperature = model, cache, temperature
        self.calls = 0  # real (uncached) calls, to check the cost budget

    def generate(self, system: str, prompt: str) -> str:
        key = prompt_hash(self.model, str(self.temperature), system, prompt)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        data = _post(f"{_BASE}/models/{self.model}:generateContent", payload)
        self.calls += 1
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(f"no candidate in model response: {str(data)[:200]}") from None
        self.cache.put(key, text)
        return text


class GeminiEmbedder:
    """Callable list[str] -> (n, d) array; per-text results cached on disk."""

    def __init__(self, model: str, cache: DiskCache, batch: int = 50):
        self.model, self.cache, self.batch = model, cache, batch

    def __call__(self, texts: list[str]) -> np.ndarray:
        keys = [prompt_hash("emb", self.model, t) for t in texts]
        out: list[list[float] | None] = [self.cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            payload = {"requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": texts[i][:8000]}]}} for i in idx]}
            data = _post(f"{_BASE}/models/{self.model}:batchEmbedContents", payload)
            for i, e in zip(idx, data["embeddings"]):
                out[i] = e["values"]
                self.cache.put(keys[i], e["values"])
        return np.asarray(out, dtype=np.float32)
