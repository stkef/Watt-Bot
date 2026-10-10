# Watt-Bot: WattBot 2026 RAG pipeline

A citation-first RAG pipeline for the [WattBot 2026](https://www.kaggle.com/) Kaggle competition. It answers questions about AI's energy, carbon and water footprint from a fixed corpus of 122 papers and reports. It returns a normalised value, the exact supporting documents, a verbatim quote, and a refusal (`is_blank`) when the corpus cannot answer.

## Pipeline

```
metadata.csv ─ fetch (once, polite) ─ parse (pages, tables→markdown) ─ chunk (page-bounded)
            ─ index (BM25 + dense, RRF, ≤4 chunks/doc) ─ answer (Gemini JSON + 3 leave-one-out shots)
            ─ normalise + map S-numbers→ref_id ─ submission.csv ─ Score.py
```

| Module | Responsibility (PRD ref) |
|---|---|
| `fetch.py` | one-time download, delay, named User-Agent, failures by id (FR-1) |
| `parse.py` | per-page text, tables as markdown, de-hyphenation, figure-page flag (FR-2, FR-3 hook) |
| `chunk.py` | ~260 words / 50 overlap, never crosses a page, tables whole (FR-4) |
| `retrieve.py` | BM25 + embeddings, RRF, per-doc cap, `doc_recall` (FR-5) |
| `answer.py` | prompt, JSON parse, citation mapping, refusals (FR-6, FR-8, FR-9) |
| `fewshot.py` | 3 most similar solved questions, never the question itself (FR-7) |
| `normalize.py` | values into the scorer's forms: number, `[lo,hi]`, `(a,b)`, text (FR-10) |
| `pipeline.py` | 4 workers, per-question trace logs, submission writer (FR-11) |
| `score_local.py` | official `Score.py` plus wrong-id lists (FR-12) |
| `llm.py` | plain REST to Gemini, temperature 0, disk cache by prompt hash |

## Usage

```bash
pip install -e ".[dev]"
# put the competition files in ./data (gitignored: the data may not be redistributed)
export GEMINI_API_KEY=...            # never commit it
wattbot fetch                         # once; second run makes zero requests
wattbot parse
wattbot index                         # --no-dense for BM25 only
wattbot recall -k 12                  # retrieval check, no LLM calls
wattbot run --split train --name baseline --limit 20
wattbot score runs/baseline/train_predictions.csv
wattbot run --split test --name baseline
```

On Kaggle, use `scripts/kaggle_notebook.py`. It reads the parsed corpus from a private dataset, so the notebook never downloads PDFs.

## Tests

`pytest -q` runs offline tests (fake model, generated PDFs, local HTTP server). With the competition data in `./data` it also checks that:

- every train ground-truth value passes through the normaliser unchanged;
- the ground truth scores exactly **1.000** against itself;
- leave-one-out holds for all 245 train questions.
