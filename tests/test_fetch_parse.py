import http.server
import threading
from functools import partial

import pandas as pd
import pytest

from wattbot.config import Config
from wattbot.fetch import fetch_all
from wattbot.parse import parse_all, load_pages, dehyphenate, table_to_markdown

reportlab = pytest.importorskip("reportlab")


def make_pdf(path, pages):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=letter)
    for lines in pages:
        y = 720
        for ln in lines:
            c.drawString(72, y, ln)
            y -= 16
        c.showPage()
    c.save()


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    make_pdf(root / "a.pdf", [["GPT-3 training used 1287 MWh."], ["Second page about water."]])
    (root / "notpdf.pdf").write_text("<html>blocked</html>" * 100)
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            hits.append(self.path)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", hits
    srv.shutdown()


def test_fetch_once_then_cached_and_failures_logged(server, tmp_path, monkeypatch):
    base, hits = server
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.delenv("HTTPS_PROXY", raising=False); monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False); monkeypatch.delenv("https_proxy", raising=False)
    cfg = Config(cache_dir=tmp_path / "cache")
    meta = pd.DataFrame({"id": ["doc_a", "doc_bad"], "url": [f"{base}/a.pdf", f"{base}/notpdf.pdf"]})
    sleeps = []
    r1 = fetch_all(meta, cfg, sleep=sleeps.append)
    assert r1["downloaded"] == ["doc_a"] and "doc_bad" in r1["failed"]
    assert sleeps == [3.0]  # polite delay between requests
    n = len(hits)
    r2 = fetch_all(meta.iloc[:1], cfg, sleep=sleeps.append)
    assert r2["cached"] == ["doc_a"] and len(hits) == n  # zero downloads on the second run

    rep = parse_all(["doc_a", "doc_missing"], cfg)
    assert rep["parsed"]["doc_a"] == 2 and "doc_missing" in rep["failed"]
    pages = load_pages(cfg, "doc_a")
    assert "1287 MWh" in pages[0]["text"] and pages[1]["page"] == 2


def test_dehyphenate_and_markdown():
    assert dehyphenate("electri-\ncity") == "electricity"
    md = table_to_markdown([["model", "MWh"], ["GPT-3", "1287"], [None, None]])
    assert md.splitlines()[0] == "| model | MWh |" and len(md.splitlines()) == 3
