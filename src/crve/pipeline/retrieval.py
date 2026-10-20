"""Retrieval of base and expanded queries with BM25 and DPH, and per-query metrics.

  patch_path_f      every pt.terrier.Retriever created afterwards runs with termpipelines=NoOp (queries are passed as
                    already analysed terms, `query_toks`)
  metrics_full      linear-gain nDCG@10/20/30/50, Recall@50/100/200/500/1000
  run_items         BM25 retrieval of (key, qid, query_toks) items; run_items_dph: the same with DPH
  expansion_items   the base query of each query and each distinct expansion (added terms at weight w = 0.20)
  live_outcomes     BM25 and DPH outcomes of the base and expanded queries (scripts/run_crve.py --compare checks them
                    against results/feature_sidecar/)

Each original query term keeps its count as query-term weight; each added term gets weight w. Retrieval depth is 1,000
documents after removing the query's excluded documents (BRIGHT exclusions, ArguAna self-match).
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

NDCG_KS = (10, 20, 30, 50)
RECALL_KS = (50, 100, 200, 500, 1000)
W = 0.20


def patch_path_f():
    import pyterrier as pt
    if not pt.java.started():
        pt.java.set_memory_limit(4096)
        pt.java.init()
    orig = pt.terrier.Retriever

    def retriever_noop(*args, **kwargs):
        controls = dict(kwargs.pop("controls", None) or {})
        controls["termpipelines"] = "NoOp"
        return orig(*args, controls=controls, **kwargs)

    pt.terrier.Retriever = retriever_noop


def metrics_full(docs, qrels) -> dict:
    ideal = sorted([v for v in qrels.values() if v >= 1], reverse=True)
    n_rel = len(ideal)
    out = {}
    for k in NDCG_KS:
        idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal[:k]))
        dcg = sum(qrels.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(docs[:k]))
        out[f"ndcg{k}"] = dcg / idcg if idcg > 0 else 0.0
    for k in RECALL_KS:
        out[f"recall{k}"] = (sum(1 for d in docs[:k] if qrels.get(d, 0) >= 1) / n_rel) if n_rel else 0.0
    return out


def run_items(ev, ds: str, items, chunk: int = 200) -> dict:
    """items: (key, qid, toks). Fetches 1,000 + |excluded| documents, removes the excluded ones, keeps 1,000."""
    by_depth = defaultdict(list)
    out = {}
    for key, qid, toks in items:
        excl = ev.queries[(ds, qid)]["excl"]
        depth = min(ev.num_docs[ds], 1000 + len(excl)) if excl else 1000
        by_depth[depth].append((key, qid, toks))
    for depth, group in sorted(by_depth.items()):
        br = ev.pt.terrier.Retriever(ev.index[ds], wmodel="BM25", num_results=depth)
        for i in range(0, len(group), chunk):
            part = group[i:i + chunk]
            res = br.transform(pd.DataFrame([{"qid": f"v{j}", "query_toks": t} for j, (_, _, t) in enumerate(part)]))
            ranked = defaultdict(list)
            if not res.empty:
                for q_id, doc in zip(res["qid"].values, res["docno"].values):
                    ranked[q_id].append(str(doc))
            for j, (key, qid, _) in enumerate(part):
                meta = ev.queries[(ds, qid)]
                docs = [d for d in ranked.get(f"v{j}", []) if d not in meta["excl"]][:1000]
                out[key] = metrics_full(docs, meta["qrels"])
    return out


def run_items_dph(ev, ds: str, items, chunk: int = 200) -> dict:
    """As run_items, with DPH."""
    by_depth = defaultdict(list)
    out = {}
    for key, qid, toks in items:
        excl = ev.queries[(ds, qid)]["excl"]
        depth = min(ev.num_docs[ds], 1000 + len(excl)) if excl else 1000
        by_depth[depth].append((key, qid, toks))
    for depth, group in sorted(by_depth.items()):
        br = ev.pt.terrier.Retriever(ev.index[ds], wmodel="DPH", num_results=depth)
        for i in range(0, len(group), chunk):
            part = group[i:i + chunk]
            res = br.transform(pd.DataFrame([{"qid": f"v{j}", "query_toks": t} for j, (_, _, t) in enumerate(part)]))
            ranked = defaultdict(list)
            if not res.empty:
                for q_id, doc in zip(res["qid"].values, res["docno"].values):
                    ranked[q_id].append(str(doc))
            for j, (key, qid, _) in enumerate(part):
                meta = ev.queries[(ds, qid)]
                docs = [d for d in ranked.get(f"v{j}", []) if d not in meta["excl"]][:1000]
                out[key] = metrics_full(docs, meta["qrels"])
    return out


def expansion_items(ev, ds, sets, w=W):
    """sets: rows (qid, arm, terms). Returns the retrieval items: each query's base query, then each distinct expansion."""
    q = sorted(sets.qid.unique())
    items, seen = [((x, ()), x, dict(ev.queries[(ds, x)]["toks"])) for x in q], {(x, ()) for x in q}
    for r in sets.itertuples():
        key = (r.qid, tuple(r.terms))
        if key in seen:
            continue
        seen.add(key)
        toks = dict(ev.queries[(ds, r.qid)]["toks"])
        for t in r.terms:
            assert t not in toks
            toks[t] = w
        items.append((key, r.qid, toks))
    return items


def live_outcomes(ds, sets, w=W):
    """BM25 and DPH outcomes of the base query and of every (qid, arm) expansion in `sets` (columns dataset, qid, arm, terms).
    Call retrieval.patch_path_f() first."""
    from crve.scoring.live_retrieval import LiveSetEvaluator
    ev = LiveSetEvaluator([ds])
    q = sorted(sets.qid.unique())
    items = expansion_items(ev, ds, sets, w)
    rows = []
    for model, fn in (("bm25", run_items), ("dph", run_items_dph)):
        res = fn(ev, ds, items)
        rows += [{"dataset": ds, "qid": r.qid, "model": model, "arm": r.arm, **res[(r.qid, tuple(r.terms))]} for r in sets.itertuples()]
        rows += [{"dataset": ds, "qid": x, "model": model, "arm": "base", **res[(x, ())]} for x in q]
    return pd.DataFrame(rows)
