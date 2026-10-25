"""The baselines of the paper: BM25, DPH, RM3 and Bo1 (fixed and tuned), BGE-QE and LLM-Q2E.

  run_harness          BM25, DPH and RM3 / Bo1 over BM25 and DPH (Terrier implementations, 10 feedback documents, 10 terms,
                       RM3 lambda = 0.5, set explicitly) over all judged queries of a collection; runs saved per pipeline
                       (the index is opened from data/cache/terrier_indices, built by scripts/build_index.py)
  run_qe               BGE-QE and LLM-Q2E over BM25 and DPH with termpipelines=NoOp; LLM calls are blocked, so a query
                       missing from the keyword cache stops the run
  per_query_from_runs  per-query metrics of the saved runs on the evaluated queries (results/queries/<c>.txt)
  grid_runs            RM3 and Bo1 on the tuning grid of Supplementary Section S1 (at most 500 queries per collection, query_subset)
  tuned_arms           per collection: default, oracle (best grid point), two-fold cross-validated choice and the
                       best-of-both oracle of RM3 / Bo1
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time

import pandas as pd

from crve.pipeline import common as C
from crve.pipeline.retrieval import metrics_full

METS = ["ndcg10", "ndcg20", "ndcg50", "recall100", "recall1000"]

HARNESS = {"bm25": ["BM25_Default", "BM25_RM3", "BM25_Bo1", "BGE_Vocab_QE", "LLM_Q2E_ZS"],
           "dph": ["DPH", "DPH_RM3", "DPH_Bo1", "DPH_BGE_Vocab_QE", "DPH_LLM_Q2E_ZS"]}
SHORT = {"BM25_Default": "BM25 (harness)", "BM25_RM3": "BM25+RM3", "BM25_Bo1": "BM25+Bo1",
         "BGE_Vocab_QE": "BM25+BGE-QE", "LLM_Q2E_ZS": "BM25+LLM-Q2E", "DPH": "DPH (harness)",
         "DPH_RM3": "DPH+RM3", "DPH_Bo1": "DPH+Bo1", "DPH_BGE_Vocab_QE": "DPH+BGE-QE",
         "DPH_LLM_Q2E_ZS": "DPH+LLM-Q2E"}
PRF_PIPELINES = ["BM25_Default", "BM25_RM3", "BM25_Bo1", "DPH", "DPH_Bo1",
                 "DPH_RM3"]
QE_PIPELINES = ["BGE_Vocab_QE", "DPH_BGE_Vocab_QE", "LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS"]

CAP = 500
DOCS, TERMS, LAMBDAS = (5, 10, 20), (10, 20), (0.3, 0.5, 0.7)
GRID = [("rm3", d, t, l) for d, t, l in itertools.product(DOCS, TERMS, LAMBDAS)] + [("bo1", d, t, None) for d, t in itertools.product(DOCS, TERMS)]
DEFAULT = {"rm3": ("rm3", 10, 10, 0.5), "bo1": ("bo1", 10, 10, None)}
WMODEL = {"bm25": "BM25", "dph": "DPH"}


# ---------------------------------------------------------------------------------------------
# fixed-setting baselines (harness runs over all judged queries; per-query metrics on the evaluated queries)
# ---------------------------------------------------------------------------------------------

def run_harness(dataset_name, pipelines, save_runs_dir, chunk_size=200, mem=3072):
    """BM25 / DPH / RM3 / Bo1 over all judged queries of the collection; saves <runs>/<c>_<pipeline>.parquet."""
    from evaluation.baselines.pyterrier_harness import PyTerrierBaselineHarness, init_pyterrier
    from evaluation.benchmark_loader import BenchmarkLoader
    init_pyterrier(mem=mem)
    import pyterrier as pt
    queries, stats = BenchmarkLoader.load_queries(dataset_name)
    index = pt.IndexFactory.of(os.path.abspath(f"data/cache/terrier_indices/{dataset_name}_default/data.properties"))
    harness = PyTerrierBaselineHarness({"index_default": index})
    return harness.run_all(queries, chunk_size=chunk_size, save_runs_dir=save_runs_dir, dataset_name=dataset_name, pipelines=pipelines)


def run_qe(datasets, runs_dir, noop=True):
    """BGE-QE and LLM-Q2E over BM25 and DPH (cached LLM keywords only); saves <runs>/<c>_<pipeline>.parquet."""
    import requests

    def blocked(*args, **kwargs):
        raise SystemExit("LLM call attempted: a query is missing from the keyword cache")

    requests.post = blocked
    import ir_measures
    from evaluation.baselines.pyterrier_harness import PyTerrierBaselineHarness, init_pyterrier
    init_pyterrier(mem=3072)
    import pyterrier as pt
    if noop:
        orig = pt.terrier.Retriever

        def retriever_noop(*args, **kwargs):
            controls = dict(kwargs.pop("controls", None) or {})
            controls["termpipelines"] = "NoOp"
            return orig(*args, controls=controls, **kwargs)

        pt.terrier.Retriever = retriever_noop
    from evaluation.baselines.pyterrier_qe import (DEFAULT_QE_CONFIG, BGEVocabQERewriter, BGEVocabSidecarManager,
                                                   LLMQEKeywordRewriter)
    from evaluation.benchmark_loader import BenchmarkLoader
    sidecar_mgr = BGEVocabSidecarManager(DEFAULT_QE_CONFIG["bge_vocab_qe"])
    llm = LLMQEKeywordRewriter(DEFAULT_QE_CONFIG["llm_q2e_zs"])
    for ds in datasets:
        index = pt.IndexFactory.of(os.path.abspath(f"data/cache/terrier_indices/{ds}_default/data.properties"))
        queries, _ = BenchmarkLoader.load_queries(ds)
        gold = {str(q["query_id"]): q.get("qrels") or {} for q in queries}
        qrels_ir = [ir_measures.Qrel(qid, str(d), int(r)) for qid, g in gold.items() for d, r in g.items()]
        harness = PyTerrierBaselineHarness(index_dict={"index_default": index})
        harness.bge_vocab_rewriter = BGEVocabQERewriter(sidecar_mgr.build_or_load_sidecar(ds, index), DEFAULT_QE_CONFIG["bge_vocab_qe"])
        harness.llm_qe_rewriter = llm
        for p in QE_PIPELINES:
            t0 = time.time()
            m = harness.evaluate_pipeline(pipeline_name=p, queries=queries, qrels_ir=qrels_ir, gold_map=gold,
                                          chunk_size=200, save_runs_dir=runs_dir, dataset_name=ds)
            print(json.dumps({"dataset": ds, "pipeline": p, "ndcg_10": m.get("ndcg_10"), "seconds": round(time.time() - t0, 1)}), flush=True)


def per_query_from_runs(ds, q_ids, run_file):
    """Per-query metrics of every harness pipeline on the evaluated queries; run_file(ds, pipeline) -> path."""
    from evaluation.benchmark_loader import BenchmarkLoader
    qs, _ = BenchmarkLoader.load_queries(ds)
    meta = {str(q["query_id"]): q for q in qs}
    rows, sources = [], {}
    for fam in HARNESS.values():
        for p in fam:
            f = run_file(ds, p)
            sources[p] = f
            run = pd.read_parquet(f, columns=["qid", "docno", "rank"])
            run = run[run.qid.isin(q_ids)].sort_values(["qid", "rank"], kind="stable")
            lists = {q: g.docno.astype(str).tolist() for q, g in run.groupby("qid", sort=False)}
            for q in q_ids:
                m = meta[q]
                excl = set(str(x) for x in m.get("excluded_doc_ids", []) or [])
                docs = [d for d in lists.get(q, []) if d not in excl][:1000]
                rows.append({"qid": q, "method": SHORT[p], **metrics_full(docs, {str(k): float(v) for k, v in m.get("qrels", {}).items()})})
            del run, lists
    df = pd.DataFrame(rows).assign(dataset=ds)
    if df.duplicated(["qid", "method"]).any() or df[METS].isna().any().any() or set(df.qid) != set(q_ids):
        raise SystemExit(f"{ds}: duplicates, missing values or query mismatch")
    return df[["dataset", "qid", "method"] + METS], sources


# ---------------------------------------------------------------------------------------------
# tuned RM3 / Bo1 (Supplementary Section S1: oracle and two-fold cross-validated choice on at most 500 queries)
# ---------------------------------------------------------------------------------------------

def cfg_name(c):
    return f"{c[0]}_d{c[1]}_t{c[2]}" + (f"_l{c[3]}" if c[3] is not None else "")


def query_subset(ds):
    q = C.queries(ds)
    return sorted(random.Random(42).sample(q, CAP)) if len(q) > CAP else q


def pipeline(pt, index, model, c, excluded_map, threads):
    """As PyTerrierBaselineHarness.build_pipeline for the RM3 / Bo1 pipelines, with the grid point's settings."""
    from evaluation.baselines.pyterrier_harness import create_exclusion_filter
    method, fb_docs, fb_terms, lam = c
    wm = WMODEL[model]
    if method == "rm3":
        rw = pt.rewrite.RM3(index, fb_terms=fb_terms, fb_docs=fb_docs, fb_lambda=lam)
    else:
        rw = pt.rewrite.Bo1QueryExpansion(index, fb_terms=fb_terms, fb_docs=fb_docs)
    if excluded_map:
        max_ex = max(len(s) for s in excluded_map.values())
        pass1 = pt.terrier.Retriever(index, wmodel=wm, num_results=min(max(100, fb_docs + max_ex), 300), threads=threads)
        pass2 = pt.terrier.Retriever(index, wmodel=wm, num_results=min(1000 + max_ex, 3000), threads=threads)
        return (pass1 >> create_exclusion_filter(excluded_map, max_docs=fb_docs) >> rw >> pass2
                >> create_exclusion_filter(excluded_map, max_docs=1000))
    pass1 = pt.terrier.Retriever(index, wmodel=wm, num_results=fb_docs, threads=threads)
    pass2 = pt.terrier.Retriever(index, wmodel=wm, num_results=1000, threads=threads)
    return pass1 >> rw >> pass2


def grid_runs(ds, which="grid", threads=1, mem=4096):
    """Per-query metrics of RM3 and Bo1 over BM25 and DPH for every grid point (or only the defaults) on query_subset(ds)."""
    from evaluation.baselines.pyterrier_harness import init_pyterrier, sanitize_default_query
    from evaluation.benchmark_loader import BenchmarkLoader
    init_pyterrier(mem=mem)
    import pyterrier as pt
    q_ids = query_subset(ds)
    qs, _ = BenchmarkLoader.load_queries(ds)
    meta = {str(q["query_id"]): q for q in qs}
    sel = [meta[x] for x in q_ids]
    # the harness built its exclusion map over all of the corpus's queries; keep that so first-pass depths match
    excluded_map = {str(q["query_id"]): set(str(d) for d in q.get("excluded_doc_ids", [])) for q in qs if q.get("excluded_doc_ids")}
    topics = pd.DataFrame([{"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])} for q in sel])
    index = pt.IndexFactory.of(os.path.abspath(f"data/cache/terrier_indices/{ds}_default/data.properties"))
    grid = list(DEFAULT.values()) if which == "default" else GRID
    rows, timing = [], {}
    for model in ("bm25", "dph"):
        for c in grid:
            t1 = time.time()
            tr = pipeline(pt, index, model, c, excluded_map, threads)
            res = pd.concat([tr.transform(topics.iloc[i:i + 200]) for i in range(0, len(topics), 200)], ignore_index=True)
            res = res.sort_values(["qid", "rank"], kind="stable")
            lists = {q: g.docno.astype(str).tolist() for q, g in res.groupby("qid", sort=False)}
            for x in q_ids:
                m = meta[x]
                excl = set(str(d) for d in m.get("excluded_doc_ids", []) or [])
                docs = [d for d in lists.get(x, []) if d not in excl][:1000]
                rows.append({"dataset": ds, "qid": x, "model": model, "method": c[0], "config": cfg_name(c), "fb_docs": c[1],
                             "fb_terms": c[2], "fb_lambda": c[3], **metrics_full(docs, {str(k): float(v) for k, v in m.get("qrels", {}).items()})})
            timing[f"{model}_{cfg_name(c)}"] = round(time.time() - t1, 1)
            print(json.dumps({"dataset": ds, "model": model, "config": cfg_name(c), "seconds": timing[f"{model}_{cfg_name(c)}"]}), flush=True)
            del res, lists
    df = pd.DataFrame(rows)
    if df.duplicated(["qid", "model", "config"]).any() or df[METS].isna().any().any():
        raise SystemExit(f"{ds}: duplicates or missing values")
    return df


def pick(per_q, qids):
    """Best grid point by mean nDCG@10 over qids; ties go to the earlier grid point."""
    m = per_q[per_q.qid.isin(qids)].groupby("config").ndcg10.mean()
    order = [cfg_name(c) for c in GRID if cfg_name(c) in m.index]
    m = m.reindex(order)
    return m.idxmax()


def tuned_arms(ds, g):
    """g: grid_runs output of one collection. Returns (per-query arms, chosen settings) for RM3 / Bo1 over BM25 and DPH."""
    rows, chosen = [], []
    q_ids = query_subset(ds)
    rng = random.Random(42)
    half = set(rng.sample(q_ids, len(q_ids) // 2))
    fold_a, fold_b = [x for x in q_ids if x in half], [x for x in q_ids if x not in half]
    for model in ("bm25", "dph"):
        for method in ("rm3", "bo1"):
            pq = g[(g.model == model) & (g.method == method)]
            rows.append(pq[pq.config == cfg_name(DEFAULT[method])][["qid"] + METS].assign(dataset=ds, model=model, arm=f"{method}_default"))
            best = pick(pq, q_ids)
            rows.append(pq[pq.config == best][["qid"] + METS].assign(dataset=ds, model=model, arm=f"{method}_oracle"))
            # two-fold cross-validated tuning: pick on one half, score the other
            ca, cb = pick(pq, fold_a), pick(pq, fold_b)
            cv = pd.concat([pq[(pq.config == cb) & pq.qid.isin(fold_a)], pq[(pq.config == ca) & pq.qid.isin(fold_b)]])
            rows.append(cv[["qid"] + METS].assign(dataset=ds, model=model, arm=f"{method}_cv"))
            chosen.append({"dataset": ds, "model": model, "method": method, "oracle": best, "cv_fold_a": ca, "cv_fold_b": cb,
                           "oracle_ndcg10": float(pq[pq.config == best].ndcg10.mean()),
                           "default_ndcg10": float(pq[pq.config == cfg_name(DEFAULT[method])].ndcg10.mean())})
        # best of both PRF methods, per corpus, on the evaluation queries
        o = {m_: pd.concat(rows).query("dataset == @ds and model == @model and arm == @m_") for m_ in ("rm3_oracle", "bo1_oracle")}
        bm = "rm3_oracle" if o["rm3_oracle"].ndcg10.mean() >= o["bo1_oracle"].ndcg10.mean() else "bo1_oracle"
        rows.append(o[bm][["qid"] + METS].assign(dataset=ds, model=model, arm="prf_oracle"))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(chosen)
