import pytest
from wattbot.data import load_train
from wattbot.normalize import normalise_value, is_blank_value
from pathlib import Path


@pytest.mark.parametrize("raw,unit,want", [
    ("1,287 MWh", "MWh", "1287"),
    ("~53%", "percent", "53"),
    ("$3.2", "USD", "3.2"),
    ("1.2 million", "tokens", "1200000"),
    ("1.2 million", "million gallons per day", "1.2"),
    ("1.5e6", "", "1500000"),
    ("10 to 50", "queries", "[10,50]"),
    ("10-50", "queries", "[10,50]"),
    ("( 1.65 , 0.5 )", "L/kWh", "(1.65,0.5)"),
    ("N/A", "", "is_blank"),
    ("Unable to answer with confidence based on the provided documents", "", "is_blank"),
    ("7.0", "", "7.0"),
    ("Flan-T5-xxl", "is_blank", "Flan-T5-xxl"),
    ("43 percent", "percent", "43"),
])
def test_cases(raw, unit, want):
    assert normalise_value(raw, unit) == want


def test_idempotent():
    for v in ["1287", "[10,50]", "(1,5)", "Alphabet", "is_blank", "0.3"]:
        assert normalise_value(normalise_value(v)) == normalise_value(v)


def test_blank_detection():
    assert is_blank_value("") and is_blank_value("NA") and not is_blank_value("0")


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not (DATA / "train_QA.csv").exists(), reason="competition data not present")
def test_train_ground_truth_unchanged():
    """FR-10: every training ground-truth value passes through normalisation unchanged."""
    tr = load_train(DATA)
    bad = [(r.id, r.answer_value, normalise_value(r.answer_value, r.answer_unit))
           for r in tr.itertuples() if normalise_value(r.answer_value, r.answer_unit) != r.answer_value]
    assert not bad, bad
