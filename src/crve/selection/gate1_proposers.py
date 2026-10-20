"""
Gate 1 candidate proposal channels of CRVE: PPMISidecarProposer (co-occurrence channel) and
SparseLexicalContextProposer (lexical profile channel), fused by RRFHybridProposer (reciprocal rank fusion, k = 60).

- Candidates are pool terms that do not occur in the analysed query.
- Ties are broken deterministically (score, then term; for the fusion: best single-channel rank, then term).
"""

from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

import pandas as pd
import pyterrier as pt

from evaluation.baselines.pyterrier_qe import get_terrier_analyzer

DEFAULT_RRF_K: int = 60


class BaseProposer:
    """Abstract base class for Gate 1 proposers."""

    def __init__(self, name: str):
        self.name = name
        self.analyzer = get_terrier_analyzer()

    def get_query_excluded_terms(self, query_text: str) -> Set[str]:
        """
        Extracts original query terms and canonical surfaces to enforce:
        P_q = P \\ AnalyzedCanonicalTerms(q).
        """
        terms, surfs = self.analyzer.analyze(query_text)
        excluded = set(terms)
        for s in surfs:
            excluded.add(str(s).lower())
            excluded.add(str(s))
        return excluded

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        """
        Proposes top_k terms for query_obj.
        Returns sorted list of (term, score) tuples.
        """
        raise NotImplementedError


class RRFHybridProposer(BaseProposer):
    """
    Reciprocal rank fusion (RRF) of the channel lists (top 500 each), k = 60.
    Deterministic tie-breaking: RRF score, then best single-channel rank, then term; the top L unique terms are kept.
    """

    def __init__(
        self,
        k: int = DEFAULT_RRF_K,
    ):
        super().__init__(name="RRFHybrid")
        self.k = k

    def fuse(
        self,
        channel_rankings: Dict[str, List[Tuple[str, float]]],
        top_l: int = 200,
    ) -> List[Tuple[str, float]]:
        """
        Fuses candidate lists from multiple proposers.
        channel_rankings: dict mapping channel_name -> list of (term, score) tuples (length <= 500).
        Returns exactly top_l unique terms sorted by RRF with deterministic tie-breaking.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        best_ranks: Dict[str, int] = {}

        for ch_name, ranking in channel_rankings.items():
            for rank_0, (term, _) in enumerate(ranking):
                rank = rank_0 + 1
                rrf_scores[term] += 1.0 / (self.k + rank)
                if term not in best_ranks or rank < best_ranks[term]:
                    best_ranks[term] = rank

        # Deterministic 3-tier tie-breaking:
        # 1. RRF score descending (-score)
        # 2. Best individual rank ascending (best_rank)
        # 3. Term string alphabetical ascending (term)
        fused = []
        for term, score in rrf_scores.items():
            b_rank = best_ranks.get(term, 999999)
            fused.append((term, score, b_rank))

        fused.sort(key=lambda x: (-x[1], x[2], x[0]))
        return [(t, s) for t, s, _ in fused[:top_l]]

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 200,
    ) -> List[Tuple[str, float]]:
        # RRF proposal requires pre-generated channel rankings
        raise NotImplementedError("Use RRFHybridProposer.fuse(channel_rankings, top_l) to fuse multiple channels.")


class PPMISidecarProposer(BaseProposer):
    """
    Co-occurrence channel: look-up of the stored top M = 600 PPMI neighbours of each query anchor
    (anchors with DF >= 2 and DF/N <= 0.12), each weighted by its normalised IDF.
    """

    def __init__(
        self,
        anchor_ppmi: Dict[str, List[Tuple[str, float]]],
        idf_map: Dict[str, float],
        df_map: Dict[str, int],
        num_docs: int,
    ):
        super().__init__(name="PPMISidecar")
        self.anchor_ppmi = anchor_ppmi
        self.idf_map = dict(idf_map)
        self.df_map = dict(df_map)
        self.num_docs = max(num_docs, 1)
        self.max_idf = max(self.idf_map.values()) if self.idf_map else 1.0

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        q_text = str(query_obj.get("question", query_obj.get("query", "")))
        excluded_terms = self.get_query_excluded_terms(q_text)

        terms, _ = self.analyzer.analyze(q_text)
        if not terms:
            return []

        unique_anchors = list(set(terms))
        scores: Dict[str, float] = defaultdict(float)

        for a in unique_anchors:
            a_df = self.df_map.get(a, 0)
            if a_df < 2 or (a_df / self.num_docs) > 0.12:
                continue

            w_a = max(self.idf_map.get(a, 0.0) / max(self.max_idf, 1e-6), 0.01)
            neighbors = self.anchor_ppmi.get(a, [])
            for cand, pmi_val in neighbors:
                if cand not in excluded_terms:
                    scores[cand] += w_a * pmi_val

        scored_terms = [(cand, score) for cand, score in scores.items()]
        scored_terms.sort(key=lambda x: (-x[1], x[0]))
        return scored_terms[:top_k]


class SparseLexicalContextProposer(BaseProposer):
    """
    Lexical profile channel: BM25 search of the analysed query over the index of the pool terms' lexical profiles.
    """

    def __init__(
        self,
        retriever: pt.terrier.Retriever,
        pool_terms: List[str],
    ):
        super().__init__(name="SparseLexicalContextProfiles")
        self.retriever = retriever
        self.pool_set = set(pool_terms)

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        q_text = str(query_obj.get("question", query_obj.get("query", "")))
        excluded_terms = self.get_query_excluded_terms(q_text)

        terms, _ = self.analyzer.analyze(q_text)
        if not terms:
            return []

        clean_query = " ".join(terms)
        q_df = pd.DataFrame([{"qid": "q", "query": clean_query}])
        res_df = self.retriever.transform(q_df)

        scored_terms = []
        for _, row in res_df.iterrows():
            t = str(row["docno"])
            if t in self.pool_set and t not in excluded_terms:
                scored_terms.append((t, float(row["score"])))

        scored_terms.sort(key=lambda x: (-x[1], x[0]))
        return scored_terms[:top_k]
