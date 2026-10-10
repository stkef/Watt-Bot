import pandas as pd
from conftest import FakeLLM, hash_embed
from wattbot.chunk import Chunk
from wattbot.config import SUBMISSION_COLUMNS
from wattbot.fewshot import FewShotPool
from wattbot.pipeline import run_questions, write_submission
from wattbot.retrieve import HybridIndex


def test_end_to_end_with_fake_model(tmp_path):
    chunks = [Chunk("p:p3:c0", "patterson2021", 3, "Training GPT-3 consumed 1,287 MWh of electricity.")]
    idx = HybridIndex.build(chunks, hash_embed)
    qs = pd.DataFrame({"id": ["q1", "q2", "q3"],
                       "question": ["Energy to train GPT-3?", "Water use of Mars base?", "Broken?"],
                       "answer_unit": ["MWh", "liters", "x"], "Cohort": ["2025"] * 3})
    llm = FakeLLM({
        "GPT-3": {"answerable": True, "answer_value": "1,287", "sources": ["S1"],
                  "quote": "consumed 1,287 MWh", "explanation": "stated"},
        "Broken": "not json",  # json.dumps of a str -> parse error -> refusal, no crash
    })
    pool = FewShotPool(pd.DataFrame({"id": ["t1"], "question": ["GPT-3 energy"], "answer": ["1"],
                                     "answer_value": ["1"], "answer_unit": ["MWh"]}))
    out = run_questions(qs, idx, llm, pool, {"patterson2021": "u"}, log_dir=tmp_path / "logs", workers=2)
    assert list(out.columns) == SUBMISSION_COLUMNS
    assert out.loc[0, "answer_value"] == "1287" and out.loc[0, "ref_id"] == "['patterson2021']"
    assert out.loc[1, "answer_value"] == "is_blank" and out.loc[2, "answer_value"] == "is_blank"
    assert (tmp_path / "logs" / "q1.json").exists()
    write_submission(out, tmp_path / "sub.csv")
    assert pd.read_csv(tmp_path / "sub.csv")["explanation"].notna().all()
