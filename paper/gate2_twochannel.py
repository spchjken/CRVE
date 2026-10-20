#!/usr/bin/env python3
"""
Table 5 of the paper (internal id T6b): what Gate 2 adds over the order of the two-channel Gate 1 list.
Read-only re-aggregation of stored result files (results/gate2_twochannel/); no retrieval.

  python paper/gate2_twochannel.py   (from the repository root; or run everything with python paper/make_all.py)

Arms, all on the same two-channel Gate 1 list (RRF, k = 60, of the co-occurrence and lexical profile channels, L = 200), w = 0.20,
all queries, S2 chunks:
  full method      (arm A in the stored files)  top K of the L candidates by Gate 2
  Gate 1 order     (arm B)                       the first K candidates of the same Gate 1 list, without Gate 2
The macro gains and the paired contribution of Gate 2 (full method minus Gate 1 order) with their 95% intervals are recomputed
from the per-query outcomes (outcomes_<collection>.parquet) and checked against the stored summary (summary.csv).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as PC
from common import OVER, RES, UNDER
import lib_full_queries as FQ  # contrast(): corpus-stratified bootstrap, 10,000 replicates, seed 42

SRC = os.path.join(RES, "gate2_twochannel")
KS = (5, 10)
TOL = 1e-12
NOTE = ("two-channel Gate 1 (RRF k = 60 of the co-occurrence and lexical profile channels, L = 200); S2 chunks; w = 0.20; "
        "base = our BM25/DPH (term pipeline disabled); all queries; 95% intervals, unadjusted; Gate 2 contribution = full method "
        "minus Gate 1 order, paired; lowered = collections more than 0.005 below the base (point estimates)")


def reading(lo, hi):
    """Above / below zero, uncertain."""
    return "above zero" if lo > 0 else "below zero" if hi < 0 else "uncertain"


def load():
    rows = []
    for ds in UNDER + OVER:
        o = pd.read_parquet(os.path.join(SRC, f"outcomes_{ds}.parquet"), columns=["dataset", "qid", "model", "arm", "K", "ndcg10"])
        o = o[o.arm.isin(["A", "B", "base"])]
        rows.append(o.assign(arm=np.where(o.arm == "base", "base", o.arm + o.K.astype(str))))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = np.where(df.dataset.isin(OVER), "over_1M", "under_1M")
    return df


def main():
    s = pd.read_csv(os.path.join(SRC, "summary.csv"))
    s = s[(s.scope == "all") & (s.metric == "ndcg10")]
    df = load()
    rows = []
    for group in ("under_1M", "over_1M"):
        for model in ("bm25", "dph"):
            sub = df[(df.group == group) & (df.model == model)]
            for k in KS:
                a, b = f"A{k}", f"B{k}"
                c = {}
                for nm, (x, y) in (("full", (a, "base")), ("gate1", (b, "base")), ("gate2", (a, b))):
                    c[nm] = FQ.contrast(sub, x, y, "ndcg10")
                    r = s[(s.group == group) & (s.model == model) & (s.contrast == f"{x} - {y}")]
                    if len(r) != 1:
                        raise SystemExit(f"summary.csv: no row for {group} {model} {x} - {y}")
                    r = r.iloc[0]
                    got = (c[nm]["macro"], c[nm]["ci95"][0], c[nm]["ci95"][1], c[nm]["below_floor"], c[nm]["n_corpora"])
                    ref = (r.macro, r.ci95_low, r.ci95_high, r["collections_below_-0.005"], r.collections)
                    if any(abs(float(u) - float(v)) > TOL for u, v in zip(got, ref)):
                        raise SystemExit(f"{group} {model} {x} - {y}: recomputed {got} differs from summary.csv {ref}")
                    c[nm]["queries"], c[nm]["level"] = int(r.queries), float(r.ref_level_ndcg10)
                base_lv = c["gate1"]["level"]
                full, g1, g2 = c["full"], c["gate1"], c["gate2"]
                rows.append({"group": PC.GROUP_NAME[group], "base_model": model.upper(), "K": k, "collections": g2["n_corpora"],
                             "queries": g2["queries"], "base_ndcg10_macro": base_lv,
                             "full_method_gain": full["macro"], "full_method_ci95_low": full["ci95"][0], "full_method_ci95_high": full["ci95"][1],
                             "full_method_gain_rel_pct": PC.rel(full["macro"], base_lv), "full_method_reading": reading(*full["ci95"]),
                             "gate1_order_gain": g1["macro"], "gate1_order_ci95_low": g1["ci95"][0], "gate1_order_ci95_high": g1["ci95"][1],
                             "gate1_order_gain_rel_pct": PC.rel(g1["macro"], base_lv), "gate1_order_reading": reading(*g1["ci95"]),
                             "gate2_contribution": g2["macro"], "gate2_contribution_ci95_low": g2["ci95"][0],
                             "gate2_contribution_ci95_high": g2["ci95"][1], "gate2_contribution_reading": reading(*g2["ci95"]),
                             "gate1_order_collections_below_-0.005": g1["below_floor"],
                             "full_method_collections_below_-0.005": full["below_floor"], "notes": NOTE})
    t6b = pd.DataFrame(rows)
    dcols = [c for c in t6b.columns if c.startswith(("full_method", "gate1_order", "gate2_contribution"))
             and not c.endswith(("_pct", "_reading", "-0.005"))]
    PC.write("T6b", t6b, delta_cols=dcols, level_cols=["base_ndcg10_macro"], pct_cols=["full_method_gain_rel_pct", "gate1_order_gain_rel_pct"],
             int_cols=["K", "collections", "queries", "gate1_order_collections_below_-0.005", "full_method_collections_below_-0.005"])
    print(t6b.drop(columns=["notes"]).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
