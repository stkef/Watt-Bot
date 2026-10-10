"""Prompting, response parsing and row assembly (FR-6, FR-8, FR-9, FR-10)."""
from __future__ import annotations

import json
import re

from .chunk import Chunk
from .citations import map_sources, ref_fields
from .config import BLANK
from .normalize import normalise_value, is_blank_value

REFUSAL_ANSWER = "Unable to answer with confidence based on the provided documents."

SYSTEM = """You answer questions about the energy, carbon and water footprint of AI, using ONLY the numbered source excerpts provided.

Rules
1. Use only the excerpts. If they do not contain what is needed to answer, set "answerable": false. Never guess; a wrong answer scores worse than a refusal.
2. "answer_value" is the bare answer: a plain number (no commas, units, %, ~, < or >), or the exact short term for a text answer. The expected unit is given; do not repeat it in answer_value. Convert to the expected unit if the source uses another.
3. If the source states a range for the value, give it as [low,high]. If the answer needs several distinct values, give them as (a,b) in the order asked.
4. If the number must be derived (e.g. combine two documents), compute it step by step in "explanation" and put the final number in answer_value. The result may appear in no excerpt.
5. If a number is a company's own claim, answer with it but say in the explanation whose claim it is.
6. If excerpts seem to conflict (different boundary, year, workload), pick the one that matches the question's wording and say why.
7. "sources" lists ONLY the excerpt labels (e.g. "S2") whose text you actually used in the quote or arithmetic. Do not cite excerpts you did not use.
8. "quote" is copied verbatim from the cited excerpts (the shortest span that supports the answer; join two spans with " ... ").
9. Reply with one JSON object and nothing else:
{"answerable": true|false, "answer": "<one sentence>", "answer_value": "<value>", "sources": ["S1"], "quote": "<verbatim>", "explanation": "<short reasoning>"}"""


def label(i: int) -> str:
    return f"S{i}"


def build_prompt(question: str, unit: str, chunks: list[Chunk], examples: list[dict]) -> tuple[str, dict[str, Chunk]]:
    by_label: dict[str, Chunk] = {}
    parts = []
    for i, ch in enumerate(chunks, start=1):
        by_label[label(i)] = ch
        parts.append(f"[{label(i)}] document={ch.doc_id} page={ch.page}\n{ch.text}")
    ex = []
    for e in examples:
        ex.append(f"Question: {e['question']}\nExpected unit: {e.get('answer_unit', '')}\n"
                  f"answer_value: {e.get('answer_value', '')}")
    prompt = (
        ("Solved examples (show the answer format only; they are different questions):\n\n"
         + "\n\n".join(ex) + "\n\n---\n\n") if ex else ""
    ) + "Source excerpts:\n\n" + "\n\n".join(parts) + (
        f"\n\n---\n\nQuestion: {question}\nExpected unit: {unit if unit else 'not given'}\n\nJSON answer:")
    return prompt, by_label


def parse_response(text: str) -> dict:
    """Tolerant JSON parse: strips code fences, takes the first {...} object."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError("no JSON object in model response")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model response is not a JSON object")
    return obj


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def locate_quote(quote: str, chunks: list[Chunk]) -> list[Chunk]:
    """Chunks that contain (any span of) the quote - used when the model forgot to list sources."""
    spans = [s.strip() for s in re.split(r"\s*\.\.\.\s*", quote or "") if len(s.strip()) > 15]
    hits = []
    for ch in chunks:
        body = _norm_space(ch.text)
        if any(_norm_space(s) in body for s in spans):
            hits.append(ch)
    return hits


def refusal_row(q: dict, reason: str) -> dict:
    unit = q.get("answer_unit", "") or BLANK
    return {
        "id": q["id"], "question": q["question"], "answer": REFUSAL_ANSWER,
        "answer_value": BLANK, "answer_unit": unit, "ref_id": BLANK, "ref_url": BLANK,
        "supporting_materials": BLANK, "explanation": reason.strip() or "No supporting evidence found in the corpus.",
        "Cohort": q.get("Cohort", ""),
    }


def build_row(q: dict, resp: dict, by_label: dict[str, Chunk], retrieved: list[Chunk],
              url_by_id: dict[str, str]) -> dict:
    """Turn a parsed model response into a submission row (or a clean refusal)."""
    unit = q.get("answer_unit", "") or ""
    explanation = str(resp.get("explanation", "")).strip()
    if not resp.get("answerable", True):
        return refusal_row(q, explanation or "The retrieved documents do not contain this information.")

    value = normalise_value(resp.get("answer_value"), unit)
    if is_blank_value(value):
        return refusal_row(q, explanation or "The model could not determine a value from the documents.")

    quote = str(resp.get("quote", "")).strip()
    doc_ids = map_sources(resp.get("sources") or [], by_label)
    if not doc_ids:  # model gave a value but no usable citation: recover from the quote, else refuse
        doc_ids = list(dict.fromkeys(c.doc_id for c in locate_quote(quote, retrieved)))
    if not doc_ids:
        return refusal_row(q, "A value was proposed but no cited document supports it, so the question is treated as unanswerable.")

    ref_id, ref_url = ref_fields(doc_ids, url_by_id)
    return {
        "id": q["id"], "question": q["question"],
        "answer": str(resp.get("answer", "")).strip() or f"{value} {unit if unit != BLANK else ''}".strip(),
        "answer_value": value, "answer_unit": unit or BLANK,
        "ref_id": ref_id, "ref_url": ref_url,
        "supporting_materials": quote or BLANK,
        "explanation": explanation or "Value taken from the cited excerpt.",
        "Cohort": q.get("Cohort", ""),
    }
