"""Gate 1: reciprocal-rank fusion (k = 60) of the PPMI and sparse lexical profile channels; top L = 200 candidates.

  candidates      the Gate 1 list of each query (each channel's top 500, restricted to pool terms not in the query, fused)
  channel_lists   the two channel lists and the fused list of each query (for the worked example of Figure 1)

The stored assets are read from data/cache/gate1 (crve.pipeline.common.GATE1_DIR); the profiles can be read from another
directory (scripts/run_crve.py --profiles-dir).

Note: the lexical profile channel queries its profile index through Terrier's standard term pipeline. Call
`candidates` before `retrieval.patch_path_f()` in the same process (patch_path_f switches every Retriever created
afterwards to termpipelines=NoOp).
"""

from __future__ import annotations

import copy
import json

import yaml

from crve.pipeline import common as C

TOP_L = 200
RRF_K = 60
GATE1_CONFIG = "configs/gate1.yaml"


def rrf(rankings: dict, top_l: int = TOP_L) -> list:
    """crve.selection.gate1_proposers.RRFHybridProposer.fuse with k = 60 (same sums, same order, same tie-breaking)."""
    from crve.selection.gate1_proposers import RRFHybridProposer
    return [t for t, _ in RRFHybridProposer(k=RRF_K).fuse(rankings, top_l=top_l)]


def g1_config():
    g1 = yaml.safe_load(open(GATE1_CONFIG, encoding="utf-8"))
    return json.loads(json.dumps(g1))


def g1_with_seed(seed):
    g1 = copy.deepcopy(g1_config())
    g1["sidecars"]["lexical_profiles"]["reservoir_seed"] = int(seed)
    return g1


def pool_file(ds):
    return C.pool_file(ds)


def pool(ds):
    return json.load(open(pool_file(ds)))["terms"]


def ppmi_dir(ds):
    return C.GATE1_DIR


def profiles_dir(ds):
    return C.GATE1_DIR


def candidates(ds, q_ids, lex_cache, lex_seed):
    """RRF(PPMI, sparse lexical profiles) top 200 for each query in q_ids; lex_cache: directory of the profiles."""
    from crve.pipeline import assets as B
    from crve.selection.gate1_proposers import PPMISidecarProposer, SparseLexicalContextProposer
    from crve.selection.gate1_sidecars import Gate1SidecarManager
    from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
    from evaluation.benchmark_loader import BenchmarkLoader
    g1 = g1_with_seed(lex_seed)
    pt = B.init_terrier(g1)
    idx, n_docs = B.open_index(pt, ds)
    df, _, idf = B.lexicon_maps(idx, n_docs)
    terms_pool = pool(ds)
    ppmi = Gate1SidecarManager(cache_dir=ppmi_dir(ds), config=g1, disallow_rebuild=True).build_or_load_bounded_ppmi_sidecar(
        ds, idx, terms_pool, num_docs=n_docs, top_m=600)
    lex = Gate1SidecarManager(cache_dir=lex_cache, config=g1, disallow_rebuild=True).build_or_load_sparse_lexical_sidecar(ds, terms_pool, num_docs=n_docs)
    pp = PPMISidecarProposer(ppmi["anchor_ppmi"], idf, df, n_docs)
    sp = SparseLexicalContextProposer(lex["retriever"], terms_pool)
    an = get_terrier_analyzer()
    pool_set = set(terms_pool)
    qs, _ = BenchmarkLoader.load_queries(ds)
    by_id = {str(q.get("query_id", q.get("qid", ""))): q for q in qs}
    out = {}
    for qid in q_ids:
        q = by_id[qid]
        terms, _ = an.analyze(str(q.get("question", q.get("query", ""))))
        excluded = set(terms)
        ch_p = [(t, s_) for t, s_ in pp.propose(q, top_k=500) if t in pool_set and t not in excluded]
        ch_s = [(t, s_) for t, s_ in sp.propose(q, top_k=500) if t in pool_set and t not in excluded]
        out[qid] = rrf({"PPMISidecar": ch_p, "SparseLexicalContextProfiles": ch_s})
    return out


def channel_lists(ds, q_ids, lex_cache, lex_seed):
    """The two channel lists of `candidates` (after the same pool and query-term filters), with their scores, and the
    fused list, per query (the worked example of Figure 1)."""
    from crve.pipeline import assets as B
    from crve.selection.gate1_proposers import PPMISidecarProposer, SparseLexicalContextProposer
    from crve.selection.gate1_sidecars import Gate1SidecarManager
    from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
    from evaluation.benchmark_loader import BenchmarkLoader
    g1 = g1_with_seed(lex_seed)
    pt = B.init_terrier(g1)
    idx, n_docs = B.open_index(pt, ds)
    df, _, idf = B.lexicon_maps(idx, n_docs)
    terms_pool = pool(ds)
    ppmi = Gate1SidecarManager(cache_dir=ppmi_dir(ds), config=g1, disallow_rebuild=True).build_or_load_bounded_ppmi_sidecar(
        ds, idx, terms_pool, num_docs=n_docs, top_m=600)
    lex = Gate1SidecarManager(cache_dir=lex_cache, config=g1, disallow_rebuild=True).build_or_load_sparse_lexical_sidecar(ds, terms_pool, num_docs=n_docs)
    pp = PPMISidecarProposer(ppmi["anchor_ppmi"], idf, df, n_docs)
    sp = SparseLexicalContextProposer(lex["retriever"], terms_pool)
    an = get_terrier_analyzer()
    pool_set = set(terms_pool)
    qs, _ = BenchmarkLoader.load_queries(ds)
    by_id = {str(q.get("query_id", q.get("qid", ""))): q for q in qs}
    out = {}
    for qid in q_ids:
        q = by_id[qid]
        terms, _ = an.analyze(str(q.get("question", q.get("query", ""))))
        excluded = set(terms)
        ch_p = [(t, s_) for t, s_ in pp.propose(q, top_k=500) if t in pool_set and t not in excluded]
        ch_s = [(t, s_) for t, s_ in sp.propose(q, top_k=500) if t in pool_set and t not in excluded]
        out[qid] = {"terms": terms, "PPMISidecar": ch_p, "SparseLexicalContextProfiles": ch_s,
                    "rrf": rrf({"PPMISidecar": ch_p, "SparseLexicalContextProfiles": ch_s})}
    return out
