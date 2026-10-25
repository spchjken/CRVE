#!/usr/bin/env python3
"""Baselines of the paper on one collection: BM25, DPH, RM3 and Bo1 (fixed and tuned), and per-query metrics.

  python scripts/run_baselines.py prf     --corpus scifact   # BM25, DPH, BM25/DPH + RM3 / Bo1 (10 docs, 10 terms, lambda 0.5)
  python scripts/run_qe_baselines.py      --corpus scifact   # BGE-QE and LLM-Q2E (separate script)
  python scripts/run_baselines.py metrics --corpus scifact [--compare results/prf_comparison/per_query_scifact.parquet]
  python scripts/run_baselines.py tuned   --corpus scifact [--compare results/prf_oracle/grid_scifact.parquet]

prf      Terrier pipelines over all judged queries of the collection, default term pipeline (the harness of the paper's
         runs); runs saved to --runs-dir/<c>_<pipeline>.parquet (default data/cache/runs).
metrics  per-query nDCG@10/20/50 and Recall@100/1000 of the ten harness pipelines (BM25, DPH, each + RM3, Bo1, BGE-QE,
         LLM-Q2E) on the evaluated queries (results/queries/<c>.txt), after removing excluded documents ->
         --out-dir/per_query_<c>.parquet (default results/runs/baselines; method names as in results/prf_comparison/).
tuned    RM3 (feedback documents 5/10/20 x terms 10/20 x lambda 0.3/0.5/0.7) and Bo1 (5/10/20 x 10/20) over BM25 and
         DPH on at most 500 evaluated queries (seed-42 sample) -> grid_<c>.parquet; per collection the oracle (best grid
         point on these queries, ties to the earlier point) and the two-fold cross-validated choice -> tuned_<c>.parquet,
         chosen_<c>.csv (Supplementary Section S1).
--compare prints the largest per-query difference against stored results and exits with status 1 if any is not zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd


def max_diff(a, b, keys, mets):
    m = a.merge(b, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"]
    return unmatched, {mt: float((m[mt] - m[mt + "_ref"]).abs().max()) for mt in mets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prf", "metrics", "tuned"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs-dir", default="data/cache/runs")
    ap.add_argument("--out-dir", default="results/runs/baselines")
    ap.add_argument("--mem", type=int, default=4096)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    from crve.pipeline import baselines as BL
    from crve.pipeline import common as C
    C.pin_terrier()
    ds = a.corpus
    os.makedirs(a.out_dir, exist_ok=True)
    if a.stage == "prf":
        res = BL.run_harness(ds, BL.PRF_PIPELINES, a.runs_dir, mem=a.mem)
        print(res[["pipeline", "ndcg_10", "recall_1000"]].to_string(index=False), flush=True)
        return
    if a.stage == "metrics":
        df, sources = BL.per_query_from_runs(ds, C.queries(ds), lambda d, p: os.path.join(a.runs_dir, f"{d}_{p}.parquet"))
        df.to_parquet(os.path.join(a.out_dir, f"per_query_{ds}.parquet"), index=False)
        print(df.groupby("method").ndcg10.mean().round(4).to_string(), flush=True)
        if a.compare:
            ref = pd.read_parquet(a.compare)
            ref = ref[ref.method.isin(set(df.method))]
            un, d = max_diff(df, ref, ["dataset", "qid", "method"], BL.METS)
            per_method = {}
            for mth, g in df.groupby("method"):
                _, dm = max_diff(g, ref[ref.method == mth], ["dataset", "qid", "method"], ["ndcg10"])
                per_method[mth] = dm["ndcg10"]
            print(json.dumps({"compare": a.compare, "unmatched_rows": un, "max_abs_diff": d, "ndcg10_max_abs_diff_by_method": per_method}, indent=1))
            if un or any(v != 0.0 for v in d.values()):
                raise SystemExit(f"{ds}: per-query metrics differ from {a.compare}")
        return
    g = BL.grid_runs(ds, mem=a.mem)
    g.to_parquet(os.path.join(a.out_dir, f"grid_{ds}.parquet"), index=False)
    arms, chosen = BL.tuned_arms(ds, g)
    arms.to_parquet(os.path.join(a.out_dir, f"tuned_{ds}.parquet"), index=False)
    chosen.to_csv(os.path.join(a.out_dir, f"chosen_{ds}.csv"), index=False)
    print(chosen.to_string(index=False), flush=True)
    if a.compare:
        ref = pd.read_parquet(a.compare)
        mets = ["ndcg10", "ndcg20", "ndcg30", "ndcg50", "recall50", "recall100", "recall200", "recall500", "recall1000"]
        un, d = max_diff(g, ref, ["dataset", "qid", "model", "config"], mets)
        print(json.dumps({"compare": a.compare, "unmatched_rows": un, "max_abs_diff": d}, indent=1))
        if un or any(v != 0.0 for v in d.values()):
            raise SystemExit(f"{ds}: grid differs from {a.compare}")


if __name__ == "__main__":
    main()
