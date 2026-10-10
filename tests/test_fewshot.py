import pandas as pd
from conftest import DATA, has_data
from wattbot.data import load_train
from wattbot.fewshot import FewShotPool

POOL = pd.DataFrame({
    "id": ["a", "b", "c", "d"],
    "question": ["energy to train GPT-3 in MWh", "energy to train GPT-3 in MWh",
                 "water use of a data center", "carbon of BLOOM training"],
    "answer": ["1287 MWh", "1287 MWh", "x", "y"], "answer_value": ["1287", "1287", "1", "2"],
    "answer_unit": ["MWh", "MWh", "L", "t"],
})


def test_never_returns_own_question_by_id_or_duplicate_text():
    shots = FewShotPool(POOL).select("a", "energy to train GPT-3 in MWh", k=3)
    assert all(s["id"] not in ("a", "b") for s in shots)


def test_ranks_by_similarity():
    shots = FewShotPool(POOL).select("zzz", "carbon emissions of BLOOM", k=1)
    assert shots[0]["id"] == "d"


@has_data
def test_leave_one_out_over_whole_train_set():
    tr = load_train(DATA)
    pool = FewShotPool(tr)
    for r in tr.itertuples():
        ids = [s["id"] for s in pool.select(r.id, r.question, k=3)]
        assert r.id not in ids and len(ids) == 3
