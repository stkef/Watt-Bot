import pytest
from wattbot.answer import build_prompt, build_row, parse_response, locate_quote
from wattbot.chunk import Chunk
from wattbot.citations import map_sources

CH = [Chunk("p21:p3:c0", "patterson2021", 3, "Training GPT-3 consumed 1,287 MWh of electricity."),
      Chunk("j25:p2:c0", "jegham2025", 2, "A US household uses 10.79 MWh per year.")]
URLS = {"patterson2021": "https://arxiv.org/pdf/2104.10350", "jegham2025": "https://arxiv.org/pdf/2505.09598"}
Q = {"id": "q9", "question": "How many household-years equal GPT-3 training?", "answer_unit": "household-years", "Cohort": "2026"}


def test_prompt_labels_and_examples():
    prompt, by_label = build_prompt(Q["question"], Q["answer_unit"], CH, [{"question": "ex q", "answer_unit": "MWh", "answer_value": "1"}])
    assert "[S1] document=patterson2021 page=3" in prompt and by_label["S2"].doc_id == "jegham2025"
    assert prompt.index("ex q") < prompt.index("Source excerpts")


def test_parse_response_tolerates_fences():
    assert parse_response('```json\n{"answerable": true}\n```')["answerable"] is True
    assert parse_response('Sure! {"a": 1} done')["a"] == 1
    with pytest.raises(ValueError):
        parse_response("no json here")


def test_map_sources_dedupes_and_drops_unknown():
    _, by = build_prompt("q", "", CH, [])
    assert map_sources(["S2", "s1", "[S2]", "S9"], by) == ["jegham2025", "patterson2021"]


def test_answer_row_combined_documents():
    _, by = build_prompt("q", "", CH, [])
    row = build_row(Q, {"answerable": True, "answer_value": "~119.3", "sources": ["S1", "S2"],
                        "quote": "consumed 1,287 MWh", "explanation": "1287/10.79 = 119.3"}, by, CH, URLS)
    assert row["answer_value"] == "119.3"
    assert row["ref_id"] == "['patterson2021', 'jegham2025']"
    assert row["ref_url"].startswith("['https://arxiv.org/pdf/2104.10350'")


def test_refusal_blanks_every_evidence_field():
    row = build_row(Q, {"answerable": False, "explanation": "not stated"}, {}, CH, URLS)
    assert all(row[c] == "is_blank" for c in ("answer_value", "ref_id", "ref_url", "supporting_materials"))
    assert row["explanation"] and row["answer_unit"] == "household-years"


def test_value_without_citation_recovers_from_quote_or_refuses():
    _, by = build_prompt("q", "", CH, [])
    row = build_row(Q, {"answer_value": "1287", "sources": [], "quote": "Training GPT-3 consumed 1,287 MWh"}, by, CH, URLS)
    assert row["ref_id"] == "['patterson2021']"
    row = build_row(Q, {"answer_value": "1287", "sources": [], "quote": "something invented by the model"}, by, CH, URLS)
    assert row["answer_value"] == "is_blank"


def test_locate_quote_multi_span():
    hits = locate_quote("Training GPT-3 consumed 1,287 MWh ... household uses 10.79 MWh per year", CH)
    assert {c.doc_id for c in hits} == {"patterson2021", "jegham2025"}
