"""
The queries and indexes of the live retrieval (crve.pipeline.retrieval): LiveSetEvaluator opens the Terrier index of each
collection (data/cache/terrier_indices/<c>_default) and holds, per query, the analysed query term counts (TerrierQueryAnalyzer),
the qrels and the excluded documents (BRIGHT exclusions, ArguAna self-match).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Sequence


class LiveSetEvaluator:
    def __init__(self, datasets: Sequence[str]) -> None:
        import pyterrier as pt
        from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
        from evaluation.benchmark_loader import BenchmarkLoader

        if not pt.java.started():
            pt.java.set_memory_limit(4096)
            pt.java.init()
        self.pt = pt
        self.analyzer = get_terrier_analyzer()
        self.index, self.num_docs, self.queries = {}, {}, {}
        for ds in datasets:
            path = os.path.abspath(f"data/cache/terrier_indices/{ds}_default/data.properties")
            self.index[ds] = pt.IndexFactory.of(path)
            self.num_docs[ds] = int(self.index[ds].getCollectionStatistics().getNumberOfDocuments())
            qs, _ = BenchmarkLoader.load_queries(ds)
            for q in qs:
                qid = str(q.get("query_id", q.get("qid", "")))
                terms, _ = self.analyzer.analyze(str(q.get("question", q.get("query", ""))))
                toks: Dict[str, float] = defaultdict(float)
                for t in terms:
                    toks[t] += 1.0
                self.queries[(ds, qid)] = {
                    "toks": dict(toks),
                    "qrels": {str(k): float(v) for k, v in q.get("qrels", {}).items()},
                    "excl": set(str(x) for x in q.get("excluded_doc_ids", q.get("exclusions", []))),
                }
