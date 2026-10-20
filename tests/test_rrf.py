"""Gate 1 fusion: RRF with k = 60 over the channels that propose a term; ties by best single-channel rank, then term."""
import pytest

import crve.selection.gate1_proposers as GP
from crve.pipeline.gate1 import RRF_K, TOP_L, rrf


@pytest.fixture(autouse=True)
def no_java(monkeypatch):
    monkeypatch.setattr(GP, "get_terrier_analyzer", lambda: None)   # the proposer base class would start Terrier


def test_rrf_scores_and_ties():
    ppmi = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    prof = [("c", 3.0), ("d", 2.0), ("a", 1.0)]
    out = rrf({"PPMISidecar": ppmi, "SparseLexicalContextProfiles": prof})
    # a: 1/61 + 1/63; c: 1/63 + 1/61 (tie with a; both best rank 1 -> by term); b: 1/62; d: 1/62 (tie, best rank 2 -> term)
    assert out == ["a", "c", "b", "d"]
    assert RRF_K == 60 and TOP_L == 200


def test_rrf_best_rank_breaks_ties_before_term():
    out = rrf({"x": [("z", 0.0), ("m", 0.0)], "y": [("q", 0.0), ("p", 0.0), ("m", 0.0)]})
    # z: 1/61, q: 1/61 -> tie, best rank 1 both -> term: q, z; m: 1/62 + 1/63 is highest
    assert out[0] == "m" and out[1:3] == ["q", "z"]


def test_rrf_top_l():
    ch = {"a": [(f"t{i:03d}", 0.0) for i in range(300)], "b": [(f"u{i:03d}", 0.0) for i in range(300)]}
    out = rrf(ch)
    assert len(out) == 200 and out[:2] == ["t000", "u000"]
    assert len(rrf(ch, top_l=10)) == 10
