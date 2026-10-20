"""Expanded query and per-query metrics: original terms keep their counts, each added term gets w = 0.20; linear-gain
nDCG and recall after removing excluded documents, depth 1,000 (+ the number of exclusions when retrieving)."""
import math

import pandas as pd
import pytest

from crve.pipeline.retrieval import W, expansion_items, metrics_full, run_items


class FakeEv:
    def __init__(self, ranking, excl=()):
        self.queries = {("ds", "q1"): {"toks": {"check": 2.0, "mail": 1.0}, "qrels": {"d1": 1.0, "d4": 2.0}, "excl": set(excl)}}
        self.num_docs = {"ds": 5000}
        self.index = {"ds": object()}
        ev = self

        class Ret:
            def __init__(self, index, wmodel, num_results):
                ev.calls.append((wmodel, num_results))

            def transform(self, df):
                rows = [{"qid": q, "docno": d} for q in df.qid for d in ranking]
                return pd.DataFrame(rows)

        class PT:
            class terrier:
                Retriever = Ret

        self.pt = PT
        self.calls = []


def test_expansion_weights():
    ev = FakeEv([])
    sets = pd.DataFrame([{"qid": "q1", "arm": "S2", "terms": ["mailbox", "postag"]}])
    items = expansion_items(ev, "ds", sets)
    assert W == 0.20
    assert items[0] == (("q1", ()), "q1", {"check": 2.0, "mail": 1.0})
    assert items[1][2] == {"check": 2.0, "mail": 1.0, "mailbox": 0.20, "postag": 0.20}
    with pytest.raises(AssertionError):                       # a query term is never added again
        expansion_items(ev, "ds", pd.DataFrame([{"qid": "q1", "arm": "S2", "terms": ["mail"]}]))


def test_effective_weight_under_bm25():
    # Terrier divides query-term weights by the largest and applies (k3 + 1) x / (k3 + x), k3 = 8
    h = lambda x: 9 * x / (8 + x)
    assert round(h(W / 1.0) / h(1.0), 2) == 0.22


def test_metrics_linear_gain():
    m = metrics_full(["d9", "d4", "d1"], {"d1": 1.0, "d4": 2.0, "d7": 0.0})
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(m["ndcg10"], dcg / idcg)
    assert m["recall50"] == 1.0 and m["recall1000"] == 1.0
    assert metrics_full([], {"d1": 1.0})["ndcg10"] == 0.0


def test_exclusions_and_depth():
    ev = FakeEv(["x", "d4", "d1"], excl=["x"])
    out = run_items(ev, "ds", [(("q1", ()), "q1", {"check": 2.0})])
    assert ev.calls == [("BM25", 1001)]                        # depth 1,000 + one excluded document
    assert out[("q1", ())]["ndcg10"] == 1.0                    # the excluded document is removed before cutting
