from conftest import hash_embed
from wattbot.chunk import Chunk
from wattbot.retrieve import HybridIndex, tokenize, rrf


def _chunks():
    out = [Chunk(f"big:p{i}:c0", "big", i, f"generic energy discussion page {i} data centers") for i in range(10)]
    out.append(Chunk("small:p1:c0", "small", 1, "Llama-3.1-405B inference on H100 emits 2.1 tCO2e"))
    return out


def test_tokenizer_keeps_exact_strings_and_parts():
    t = tokenize("Llama-3.1-405B on H100 tCO2e")
    assert "llama-3.1-405b" in t and "405b" in t and "h100" in t and "tco2e" in t


def test_bm25_finds_exact_model_name():
    idx = HybridIndex.build(_chunks(), None)
    assert idx.search("emissions of Llama-3.1-405B", top_k=1)[0].doc_id == "small"


def test_per_document_cap():
    idx = HybridIndex.build(_chunks(), hash_embed)
    res = idx.search("energy data centers H100", top_k=12, per_doc=4)
    assert sum(c.doc_id == "big" for c in res) == 4 and any(c.doc_id == "small" for c in res)


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [2, 1, 3]])[0][0] in (1, 2)
    assert rrf([[5, 1], [1, 6]])[0][0] == 1


def test_doc_recall():
    idx = HybridIndex.build(_chunks(), hash_embed)
    assert idx.doc_recall("Llama-3.1-405B H100 tCO2e", {"small"}) == 1.0
