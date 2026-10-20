#!/usr/bin/env python3
"""Run CRVE on one collection: Gate 1, Gate 2, retrieval of the base and expanded queries (BM25 and DPH), evaluation.

  python scripts/run_crve.py --corpus scifact
  python scripts/run_crve.py --corpus scifact --compare results/feature_sidecar/outcomes_scifact.parquet

CRVE as in the paper: pool of 10,000 terms; Gate 1 = RRF (k = 60) of the PPMI and sparse lexical profile channels (top 500
each), L = 200 candidates; Gate 2 = maximum cosine between the query (BGE-small-en-v1.5, CPU) and each candidate's stored
S2 chunks; the top K = 5 terms are added at weight w = 0.20; BM25 and DPH with termpipelines=NoOp, 1,000 documents.

Inputs: data/cache/terrier_indices/<c>_default, data/cache/gate1/<c>_{canonical_pool.json, bounded_ppmi.parquet,
lexical_profiles.parquet, lexical_profiles_idx} (--profiles-dir to use other profiles), data/cache/sidecars/S2/<c>
(--sidecar-root), the evaluated queries results/queries/<c>.txt.

Outputs (--out-dir, default results/runs/crve): gate1_<c>.parquet (qid, the L candidates in Gate 1 order),
sets_<c>.parquet (qid, arm S2, the K added terms), outcomes_<c>.parquet (dataset, qid, model, arm in {S2, base}, nDCG@10/20/30/50,
Recall@50/100/200/500/1000; the columns of results/feature_sidecar/outcomes_<c>.parquet).
--compare <stored outcomes> prints the largest per-query difference for each model and arm and exits with status 1 if
any is not zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd


def compare(out, ref_path, models=("bm25", "dph"), arms=("S2", "base")):
    from crve.pipeline.common import METRICS
    ref = pd.read_parquet(ref_path)
    res, ok = {}, True
    for model in models:
        for arm in arms:
            a = out[(out.model == model) & (out.arm == arm)].set_index("qid")[METRICS].sort_index()
            b = ref[(ref.model == model) & (ref.arm == arm)].set_index("qid")[METRICS].sort_index()
            same_q = a.index.equals(b.index)
            d = float((a - b.reindex(a.index)).abs().max().max()) if same_q else float("nan")
            res[f"{model}/{arm}"] = {"queries": int(len(a)), "same_queries": bool(same_q), "max_abs_diff": d,
                                     "macro_ndcg10": float(a.ndcg10.mean()), "stored_macro_ndcg10": float(b.ndcg10.mean())}
            ok = ok and same_q and d == 0.0
    return res, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--profiles-dir", default=None, help="directory with <c>_lexical_profiles.parquet and _idx (default data/cache/gate1)")
    ap.add_argument("--profile-seed", type=int, default=42)
    ap.add_argument("--sidecar-root", default=None, help="directory with <c>/ S2 sidecars (default data/cache/sidecars/S2)")
    ap.add_argument("--out-dir", default="results/runs/crve")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    from crve.pipeline import common as C
    from crve.pipeline import gate1 as G
    from crve.pipeline import gate2 as G2
    from crve.pipeline import retrieval as R
    C.pin_terrier()
    ds = a.corpus
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    q = C.queries(ds)
    # Gate 1 (before patch_path_f: the profile channel runs through Terrier's standard term pipeline)
    cands = G.candidates(ds, q, a.profiles_dir or G.profiles_dir(ds), a.profile_seed)
    pd.DataFrame([{"dataset": ds, "qid": x, "terms": list(cands[x])} for x in q]).to_parquet(os.path.join(a.out_dir, f"gate1_{ds}.parquet"), index=False)
    t1 = time.time()
    # Gate 2
    t5 = G2.top5(ds, cands, C.sidecar_dir(ds, a.sidecar_root or C.SIDECAR_DIR))
    sets = pd.DataFrame([{"dataset": ds, "qid": x, "arm": "S2", "terms": list(t5[x])} for x in q])
    sets.to_parquet(os.path.join(a.out_dir, f"sets_{ds}.parquet"), index=False)
    t2 = time.time()
    # retrieval and evaluation
    R.patch_path_f()
    out = R.live_outcomes(ds, sets)
    out.to_parquet(os.path.join(a.out_dir, f"outcomes_{ds}.parquet"), index=False)
    t3 = time.time()
    summ = out.groupby(["model", "arm"]).ndcg10.mean().unstack()
    info = {"corpus": ds, "queries": len(q), "seconds": {"gate1": round(t1 - t0, 1), "gate2": round(t2 - t1, 1), "retrieval": round(t3 - t2, 1)},
            "ndcg10": {m: {k: round(float(v), 4) for k, v in summ.loc[m].items()} for m in summ.index}}
    print(json.dumps(info), flush=True)
    if a.compare:
        res, ok = compare(out, a.compare)
        print(json.dumps({"compare": a.compare, "result": res}, indent=1), flush=True)
        if not ok:
            raise SystemExit(f"{ds}: outcomes differ from {a.compare}")


if __name__ == "__main__":
    main()
