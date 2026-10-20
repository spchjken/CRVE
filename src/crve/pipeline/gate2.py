"""Gate 2: order the Gate 1 candidates by the maximum cosine between the query and each candidate's stored chunks.

top5 keeps the first K = 5 candidates of each Gate 1 list in Gate 2 order. The query is encoded on CPU with
BAAI/bge-small-en-v1.5 at revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (CLS pooling, unit length, at most 128 tokens,
no instruction prefix: crve.scoring.context_evidence.encode_queries). Scores are (cos + 1) / 2, which orders like the
cosine; ties go to the better Gate 1 rank, then to the term (crve.scoring.ranking.ranked_list); a candidate without
stored chunks is placed last.
"""

from __future__ import annotations

import numpy as np

K = 5


def top5(ds, lists, sidecar_dir):
    from crve.selection.gate2_sidecar import ContextSidecar
    from crve.scoring.context_evidence import encode_queries, score_query_terms
    from crve.scoring.ranking import ranked_list
    from evaluation.benchmark_loader import BenchmarkLoader
    qtext = BenchmarkLoader.load_query_texts_only(ds)
    qv = encode_queries({(ds, q): qtext[q] for q in lists}, device="cpu", seed=42)
    sc = ContextSidecar(sidecar_dir, verify_checksums=True)
    out = {}
    for qid, terms in lists.items():
        top1 = {}
        if terms:
            res, _ = score_query_terms(sc, qv[(ds, qid)], terms)
            top1 = {x["term"]: (max(x["sims"]) if x["sims"] else np.nan) for x in res}
        out[qid] = ranked_list(terms, [top1.get(t, np.nan) for t in terms], np.arange(1, len(terms) + 1, dtype=float), top=K)
    return out
