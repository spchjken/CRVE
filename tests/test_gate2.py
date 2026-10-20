"""Gate 2: score = max over the term's stored chunks of (cos + 1) / 2; order by score, then Gate 1 rank, then term;
terms without stored chunks last; keep K = 5."""
import numpy as np

from crve.pipeline.gate2 import K
from crve.scoring.context_evidence import score_query_terms
from crve.scoring.ranking import ranked_list


class FakeSidecar:
    def __init__(self, emb, terms):
        self.embeddings = np.asarray(emb, dtype=np.float16)
        self.terms = terms

    def get_term_metadata(self, t):
        if t not in self.terms:
            return None
        ci = self.terms[t]
        return {"status": "VERIFIED" if ci else "UNVERIFIED", "sample_n": len(ci), "population_N": 2 * len(ci),
                "distinct_docs_S": len(ci), "chunk_indices": ci}


def test_max_over_chunks():
    q = np.array([1.0, 0.0], dtype=np.float32)
    emb = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]]
    sc = FakeSidecar(emb, {"a": [1, 2], "b": [3], "c": [0, 3], "d": []})
    rows, pool = score_query_terms(sc, q, ["a", "b", "c", "d", "zz"])
    top = {r["term"]: (max(r["sims"]) if r["sims"] else np.nan) for r in rows}
    assert np.isclose(top["a"], (0.6 + 1) / 2, atol=1e-3)    # best of cos 0.0 and 0.6
    assert np.isclose(top["b"], 0.0) and np.isclose(top["c"], 1.0)
    assert np.isnan(top["d"]) and np.isnan(top["zz"])         # unverified / not in the sidecar: no score
    assert len(pool) == 4


def test_order_and_tie_breaks():
    terms = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    scores = [0.7, np.nan, 0.9, 0.7, 0.8, 0.7, 0.1]
    gate1 = np.arange(1, 8, dtype=float)
    out = ranked_list(terms, scores, gate1, top=K)
    assert K == 5
    assert out == ["t3", "t5", "t1", "t4", "t6"]               # equal scores: better Gate 1 rank first
    full = ranked_list(terms, scores, gate1, top=10)
    assert full[-1] == "t2"                                    # no stored chunks: last, not removed
    assert ranked_list(["b", "a"], [0.5, 0.5], [np.nan, np.nan]) == ["a", "b"]   # then by term
