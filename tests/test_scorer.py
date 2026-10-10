from conftest import DATA, has_data
from wattbot.data import load_train
from wattbot.score_local import score_predictions


@has_data
def test_ground_truth_scores_exactly_one():
    tr = load_train(DATA)
    res = score_predictions(tr, tr.copy(), DATA, verbose=False)
    assert res["overall"] == 1.0
    assert res["wrong_value"] == [] and res["wrong_ref"] == []


@has_data
def test_refusing_everything_is_bad_and_lists_wrong_ids():
    tr = load_train(DATA)
    sub = tr.copy()
    for c in ("answer_value", "ref_id", "ref_url", "supporting_materials"):
        sub[c] = "is_blank"
    res = score_predictions(tr, sub, DATA, verbose=False)
    assert res["overall"] < 0.1
    assert len(res["wrong_value"]) == (tr["is_NA"] != "1").sum()
