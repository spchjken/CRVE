"""Terrier index access, vocabulary pool and the restricted (dense) PPMI build.

  init_terrier, open_index, lexicon_maps   JVM start (4 GiB heap), the collection's Terrier index, DF / CF / IDF maps
  build_pool_v2, pool_sha, pool_record     the pool of P = 10,000 terms and its JSON record
  query_anchors, dense_ppmi, save_ppmi     PPMI lists restricted to the anchors of the evaluated queries, with a dense count
                                           array (the five collections over 1M documents)

The full PPMI build for collections under 1M documents and the sparse lexical profiles are built by
crve.selection.gate1_sidecars.Gate1SidecarManager (build_or_load_bounded_ppmi_sidecar / build_or_load_sparse_lexical_sidecar).
"""

from __future__ import annotations

import hashlib
import math
import os
import time

import numpy as np


def init_terrier(g1):
    import pyterrier as pt
    if not pt.java.started():
        pt.java.set_memory_limit(int(g1.get("memory_watchdog", {}).get("jvm_heap_gib", 4)) * 1024)
        pt.java.init()
    return pt


def open_index(pt, ds):
    idx = pt.IndexFactory.of(os.path.abspath(f"data/cache/terrier_indices/{ds}_default/data.properties"))
    return idx, int(idx.getCollectionStatistics().getNumberOfDocuments())


def lexicon_maps(idx, n_docs):
    df, cf, idf = {}, {}, {}
    for e in idx.getLexicon():
        t = str(e.getKey())
        d, c = int(e.getValue().getDocumentFrequency()), int(e.getValue().getFrequency())
        df[t], cf[t] = d, c
        idf[t] = max(math.log(1.0 + (n_docs - d + 0.5) / (d + 0.5)), 0.0)
    return df, cf, idf


def build_pool_v2(df, cf, idf, n_docs, size):
    from evaluation.pool_generators import compute_salience_score, is_eligible_term
    el = [t for t in df if is_eligible_term(t, df[t], cf[t], n_docs)]
    ranked = sorted(el, key=lambda t: (-compute_salience_score(idf[t], df[t]), t))[:size]
    return ranked, len(el)


def pool_sha(terms):
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


def pool_record(ds, terms, n_el):
    """The pool file's content (<c>_canonical_pool.json)."""
    return {"dataset": ds, "pool_size": len(terms), "sha256": pool_sha(terms), "eligible": n_el,
            "rule": "pool_generators.is_eligible_term; top by IDF*ln(1+DF), Lucene IDF, ties by term",
            "terms": terms}


def query_anchors(ds: str, qids) -> set:
    from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
    from evaluation.benchmark_loader import BenchmarkLoader
    analyzer = get_terrier_analyzer()
    qs, _ = BenchmarkLoader.load_queries(ds)
    by = {str(q.get("query_id", q.get("qid", ""))): q for q in qs}
    out = set()
    for q in qids:
        terms, _ = analyzer.analyze(str(by[q].get("question", by[q].get("query", ""))))
        out |= set(terms)
    return out


def dense_ppmi(ds: str, pool_terms, anchors, num_docs: int, index, top_m: int = 600):
    """Bounded PPMI restricted to `anchors`, as gate1_sidecars.build_or_load_bounded_ppmi_sidecar, with a dense count array."""
    from crve.selection import gate1_sidecars as GSC
    from crve.selection.gate1_sidecars import check_build_watchdog
    from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
    from evaluation.benchmark_loader import BenchmarkLoader
    t0 = time.perf_counter()
    analyzer = get_terrier_analyzer()
    df_map = {}
    for entry in index.getLexicon():
        df_map[str(entry.getKey())] = int(entry.getValue().getDocumentFrequency())
    eligible = sorted(t for t in anchors if df_map.get(t, 0) >= 2 and df_map[t] / num_docs <= 0.12)
    A, P = len(eligible), len(pool_terms)
    if A * P * 4 > 3 * 1024 ** 3:
        raise SystemExit(f"dense PPMI array too large: {A} anchors x {P} pool terms")
    a_idx = {t: i for i, t in enumerate(eligible)}
    p_idx = {t: j for j, t in enumerate(pool_terms)}
    M = np.zeros(A * P, dtype=np.uint32)
    n_docs = 0
    for _, text in BenchmarkLoader.stream_corpus(ds):
        n_docs += 1
        if n_docs % 2000 == 0:
            check_build_watchdog(12.0)
        if n_docs % 500000 == 0:
            print(f"[ppmi {ds}] {n_docs} docs {time.perf_counter() - t0:.0f}s", flush=True)
        terms, _ = analyzer.analyze(text)
        doc_terms = set(terms)
        pj = [p_idx[t] for t in doc_terms if t in p_idx]
        if not pj:
            continue
        ai = [a_idx[t] for t in doc_terms if t in a_idx]
        if not ai:
            continue
        M[(np.asarray(ai, dtype=np.int64)[:, None] * P + np.asarray(pj, dtype=np.int64)[None, :]).ravel()] += 1
    M = M.reshape(A, P)
    pool_df = np.array([df_map.get(t, 0) for t in pool_terms], dtype=np.int64)
    anchor_ppmi, peak_pairs = {}, 0
    for i, a in enumerate(eligible):
        row = M[i]
        nz = np.nonzero(row)[0]
        if a in p_idx:
            nz = nz[nz != p_idx[a]]          # the library never counts an anchor with itself
        peak_pairs += len(nz)
        js = nz[(row[nz] >= 2) & (pool_df[nz] >= 1)]
        if not len(js):
            continue
        ratio = (num_docs * row[js].astype(np.int64)) / (df_map[a] * pool_df[js])   # exact as Python int / int below 2**53
        scored = [(pool_terms[j], math.log(x)) for j, x in zip(js.tolist(), ratio.tolist())]
        scored = [(t, p) for t, p in scored if p > 0.0]
        if scored:
            scored.sort(key=lambda x: (-x[1], x[0]))
            anchor_ppmi[a] = scored[:top_m]
    meta = {"dataset": ds, "pool_sha256": GSC.compute_pool_sha256(pool_terms), "pool_size": len(pool_terms), "num_docs": num_docs,
            "analyzer_version": GSC.ANALYZER_VERSION, "top_m": top_m, "num_anchors": len(anchor_ppmi), "peak_pairs": int(peak_pairs),
            "corpus_source_hash": GSC.compute_corpus_source_hash(ds), "lexicon_semantic_hash": GSC.compute_lexicon_semantic_hash(index),
            "elapsed_sec": round(time.perf_counter() - t0, 2), "anchor_subset_size": len(anchors), "anchor_subset_eligible": A,
            "anchor_subset_sha256": hashlib.sha256("\n".join(sorted(anchors)).encode()).hexdigest(),
            "note": "restricted to the anchors of the evaluated queries, dense counts; exact for those anchors only",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return anchor_ppmi, meta


def save_ppmi(path: str, anchor_ppmi, meta) -> None:
    from crve.selection import gate1_sidecars as GSC
    mgr = GSC.Gate1SidecarManager.__new__(GSC.Gate1SidecarManager)
    mgr.disk_caps = dict(GSC.DEFAULT_MAX_DISK_MB)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    GSC.Gate1SidecarManager._save_ppmi_parquet(mgr, path, anchor_ppmi, meta)
