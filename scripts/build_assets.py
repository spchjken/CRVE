#!/usr/bin/env python3
"""Build the Gate 1 assets of a collection: vocabulary pool, PPMI co-occurrence lists, sparse lexical profiles.

  python scripts/build_assets.py pool     --corpus scifact
  python scripts/build_assets.py ppmi     --corpus scifact
  PYTHONHASHSEED=42 python scripts/build_assets.py profiles --corpus scifact

Writes into --out-dir (default data/cache/gate1); needs the Terrier index (scripts/build_index.py) and the raw data.

pool      P = 10,000 terms (all eligible terms if fewer): eligible = DF >= 2, CF >= 3, length >= 2, not all digits and
          DF/N <= 0.15 (evaluation.pool_generators.is_eligible_term); ranked by IDF x ln(1 + DF) with the Lucene IDF
          ln(1 + (N - DF + 0.5) / (DF + 0.5)), ties by term. -> <c>_canonical_pool.json
ppmi      anchors: lexicon terms with DF >= 2 and DF/N <= 0.12; pairs with DF(a, t) >= 2; PPMI = max(0, ln(N DF(a,t) /
          (DF(a) DF(t)))); M = 600 pool terms kept per anchor. Under 1M documents: all anchors
          (Gate1SidecarManager.build_or_load_bounded_ppmi_sidecar). Over 1M: only the anchors of the evaluated queries
          (results/queries/<c>.txt), with a dense count array (crve.pipeline.assets.dense_ppmi); for those anchors the
          lists equal the full computation. -> <c>_bounded_ppmi.parquet
profiles  per pool term up to R = 50 windows sampled by reservoir sampling over its occurrences, each with up to 32
          analysed terms before and 31 after the occurrence (the centre occurrence of the term removed), indexed with
          Terrier as a small document set. -> <c>_lexical_profiles.parquet and <c>_lexical_profiles_idx/
          The sampler uses one random.Random(42) shared across terms and visits each document's pool terms in Python set
          order, so the profiles depend on the string hash seed: run with PYTHONHASHSEED=42 (required). The profiles used
          for the paper were built without a recorded hash seed and cannot be rebuilt exactly; use the stored profiles to
          reproduce the paper's numbers (data/README.md).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "ppmi", "profiles"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", default="data/cache/gate1")
    ap.add_argument("--pool-dir", default=None, help="where to read the pool from (default: --out-dir)")
    ap.add_argument("--seed", type=int, default=42, help="reservoir seed of the profiles (the paper: 42)")
    a = ap.parse_args()
    from crve.pipeline import assets as B
    from crve.pipeline import common as C
    from crve.pipeline import gate1 as G
    ds = a.corpus
    if a.stage == "profiles" and os.environ.get("PYTHONHASHSEED") != str(a.seed):
        raise SystemExit(f"run with PYTHONHASHSEED={a.seed}")
    C.pin_terrier()
    t0 = time.time()
    g1 = G.g1_with_seed(a.seed)
    pt = B.init_terrier(g1)
    idx, n = B.open_index(pt, ds)
    os.makedirs(a.out_dir, exist_ok=True)
    pool_path = os.path.join(a.pool_dir or a.out_dir, f"{ds}_canonical_pool.json")
    if a.stage == "pool":
        df, cf, idf = B.lexicon_maps(idx, n)
        terms, n_el = B.build_pool_v2(df, cf, idf, n, 10000)
        json.dump(B.pool_record(ds, terms, n_el), open(os.path.join(a.out_dir, f"{ds}_canonical_pool.json"), "w"))
        info = {"docs": n, "eligible": n_el, "pool": len(terms), "sha256": B.pool_sha(terms)}
        del df, cf, idf
        gc.collect()
    elif a.stage == "ppmi":
        from crve.selection.gate1_sidecars import Gate1SidecarManager
        pool = json.load(open(pool_path))["terms"]
        if ds in C.OVER_1M:
            anchors = B.query_anchors(ds, C.queries(ds))
            mine, meta = B.dense_ppmi(ds, pool, anchors, n, idx)
            B.save_ppmi(os.path.join(a.out_dir, f"{ds}_bounded_ppmi.parquet"), mine, meta)
        else:
            meta = Gate1SidecarManager(cache_dir=a.out_dir, config=g1).build_or_load_bounded_ppmi_sidecar(
                ds, idx, pool, num_docs=n, top_m=600)["metadata"]
        info = {k: meta.get(k) for k in ("num_anchors", "peak_pairs", "anchor_subset_size", "anchor_subset_eligible", "elapsed_sec")}
    else:
        from crve.selection.gate1_sidecars import Gate1SidecarManager
        pool = json.load(open(pool_path))["terms"]
        r = Gate1SidecarManager(cache_dir=a.out_dir, config=g1).build_or_load_sparse_lexical_sidecar(ds, pool, num_docs=n)
        info = {"seconds": r.get("timing_s"), "reservoir_seed": r["metadata"]["reservoir_seed"]}
    print(json.dumps({"stage": a.stage, "corpus": ds, "out_dir": a.out_dir, **info, "elapsed_seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
