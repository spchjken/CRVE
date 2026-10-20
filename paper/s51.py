#!/usr/bin/env python3
"""
Section 5.1 (effectiveness): Tables 3 and A3 (T4), the per-collection gains of Figures 2 and 3 (F1_data), T3b; the Gate 1
channel numbers of Section 5.3 (T3).
Read-only re-aggregation of stored result files; no retrieval.

  python paper/s51.py        (from the repository root; or run everything with python paper/make_all.py)

Writes paper/output/{T3,T3b,F1_data,T4}.csv (+ _rounded) and checks the percentile in the caption of Figure 1.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import GROUP_NAME, NAME, OVER, RES, UNDER, boot_ci, group_of, lvl, rel, write
import lib_full_queries as FQ  # contrast(): corpus-stratified bootstrap, 10,000 replicates, seed 42

SL = RES
RC = f"{SL}/recheck"
FLOOR = 0.005
REPS, SEED = 10000, 42
CHUNKS = "randomly drawn, 20 per term (seed 42)"
GATE1 = {"primary_flat": "two-channel (co-occurrence and lexical profile)", "boost_flat": "co-occurrence channel alone"}


# ---------------------------------------------------------------- helpers

def boot_ci_large(x, reps=REPS, seed=SEED, chunk=500):
    """Same draws and result as common.boot_ci, generated in row chunks to bound memory for large query sets."""
    x = np.asarray(x, dtype=float)
    if len(x) * reps <= 20_000_000:
        return boot_ci(x, reps, seed)
    rng = np.random.default_rng(seed)
    m = np.empty(reps)
    for s in range(0, reps, chunk):
        k = min(chunk, reps - s)
        m[s:s + k] = x[rng.integers(0, len(x), size=(k, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


T4_METS = ["ndcg10", "ndcg50", "recall1000"]
DIRECTION = {"above 0": "above zero", "below 0": "below zero", "undecided": "uncertain"}  # a_holm.json wording -> paper wording


def zero_reading(lo, hi):
    """95% (or Holm) interval against zero."""
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "uncertain")


def check(label, a, b, tol=1e-9):
    if abs(a - b) > tol:
        raise SystemExit(f"cross-check failed: {label}: {a} vs {b}")


# ---------------------------------------------------------------- T3 (Section 5.3, Gate 1 channels)

def t3():
    """Gate 1 channels on the 30,541 held-out queries of eleven BEIR collections (randomly drawn chunks, BM25): the gains of the
    two-channel Gate 1 and of the co-occurrence channel alone (after the evaluation corrections, results/recheck/full_queries/),
    and the paired difference from adding the lexical profile channel (measured before the corrections, results/full_queries/)."""
    v = json.load(open(f"{RC}/full_queries/verdict.json"))["primaries"]
    o = pd.read_parquet(f"{RC}/full_queries/outcomes.parquet", columns=["dataset", "qid", "arm", "group", "ndcg10"])
    rows = []
    for pid in ("P1", "P2", "P3", "P4"):
        p = v[pid]
        a, b = [s.strip() for s in p["contrast"].split(" - ")]
        sub = o[o.group == p["group"]]
        piv = sub.pivot_table(index=["dataset", "qid"], columns="arm", values="ndcg10", aggfunc="first")
        d = (piv[a] - piv[b])
        cm = d.groupby(level=0).mean()
        macro = float(cm.mean())
        check(f"{pid} macro", macro, p["macro"])
        bm25 = float(piv[b].groupby(level=0).mean().mean())
        rows.append({"quantity": "gain over BM25", "gate1": GATE1[a], "group": GROUP_NAME[p["group"]], "evaluation": "after the corrections",
                     "chunks": CHUNKS, "collections": int(p["n_corpora"]), "queries": int(len(d)), "bm25_ndcg10_macro": bm25,
                     "value": macro, "relative_pct": rel(macro, bm25), "ci95_low": np.nan, "ci95_high": np.nan, "reading": ""})
    del o
    # paired difference two-channel minus co-occurrence alone, before the corrections (stored as the reverse contrast)
    o = pd.read_parquet(f"{SL}/full_queries/outcomes.parquet", columns=["dataset", "qid", "arm", "group", "ndcg10"])
    summ = pd.read_csv(f"{SL}/full_queries/summary.csv")
    for g in ("under_1M", "over_1M"):
        sub = o[(o.group == g) & o.arm.isin(["primary_flat", "boost_flat"])]
        c = FQ.contrast(sub, "primary_flat", "boost_flat", "ndcg10")
        s = summ[(summ.scope == g) & (summ.metric == "ndcg10") & (summ.contrast == "boost_flat - primary_flat")].iloc[0]
        check(f"paired {g} macro", c["macro"], -float(s.macro), 1e-12)
        check(f"paired {g} ci95 low", c["ci95"][0], -float(s.ci95_high), 1e-12)
        check(f"paired {g} ci95 high", c["ci95"][1], -float(s.ci95_low), 1e-12)
        nq = int(sub.drop_duplicates(["dataset", "qid"]).shape[0])
        rows.append({"quantity": "paired difference: two-channel minus co-occurrence alone", "gate1": "",
                     "group": GROUP_NAME[g], "evaluation": "before the corrections", "chunks": CHUNKS, "collections": c["n_corpora"],
                     "queries": nq, "bm25_ndcg10_macro": np.nan, "value": c["macro"], "relative_pct": np.nan,
                     "ci95_low": c["ci95"][0], "ci95_high": c["ci95"][1], "reading": zero_reading(*c["ci95"])})
    del o
    df = pd.DataFrame(rows)
    write("T3", df, delta_cols=["value", "ci95_low", "ci95_high"], level_cols=["bm25_ndcg10_macro"], pct_cols=["relative_pct"],
          int_cols=["collections", "queries"])
    return df


# ---------------------------------------------------------------- Figure 1 caption

def figure1_typicality():
    """Figure 1 caption: the example's gain (FiQA q_beir_fiqa_2713, CRVE (S2) over BM25) lies at the 47th percentile of the 792
    held-out queries that CRVE improves on the six BEIR collections under 1M with held-out queries (share of these queries with a
    strictly smaller gain)."""
    q = pd.read_csv(f"{SL}/full_queries/queries.csv")
    q = q[q.role == "fresh"]
    gains = []
    for ds in ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "quora"):
        o = pd.read_parquet(f"{SL}/feature_sidecar/outcomes_{ds}.parquet", columns=["qid", "model", "arm", "ndcg10"])
        p = o[(o.model == "bm25") & o.arm.isin(["S2", "base"])].pivot_table(index="qid", columns="arm", values="ndcg10", aggfunc="first")
        d = (p["S2"] - p["base"])
        d = d[d.index.isin(set(q[q.dataset == ds].qid))]
        gains.append(d[d > 0])
    g = pd.concat(gains)
    x = float(g.loc["q_beir_fiqa_2713"])
    pct = 100.0 * float((g < x).mean())
    print(f"Figure 1: gain {x:+.4f} at the {pct:.1f}th percentile of {len(g)} improved held-out queries", flush=True)
    check("Figure 1 improved queries", len(g), 792, 0)
    check("Figure 1 percentile", round(pct), 47, 0)


# ---------------------------------------------------------------- F1 and T3b

def f1_data():
    rows = []
    for ds in UNDER + OVER:
        o = pd.read_parquet(f"{SL}/feature_sidecar/outcomes_{ds}.parquet", columns=["qid", "model", "arm", "ndcg10"])
        for model in ("bm25", "dph"):
            m = o[o.model == model]
            p = m[m.arm.isin(["S2", "base"])].pivot_table(index="qid", columns="arm", values="ndcg10", aggfunc="first")
            if p.isna().any().any():
                raise SystemExit(f"{ds} {model}: S2 and base query sets differ")
            d = (p["S2"] - p["base"]).to_numpy()
            lo, hi = boot_ci_large(d)
            rows.append({"dataset": ds, "collection": NAME[ds], "group": group_of(ds), "base": model.upper(), "queries": len(d),
                         "base_ndcg10": float(p["base"].mean()), "crve_s2_ndcg10": float(p["S2"].mean()), "gain": float(d.mean()),
                         "ci_low": lo, "ci_high": hi, "relative_gain_pct": rel(float(d.mean()), float(p["base"].mean()))})
        del o
    df = pd.DataFrame(rows)
    write("F1_data", df, delta_cols=["gain", "ci_low", "ci_high"], level_cols=["base_ndcg10", "crve_s2_ndcg10"],
          pct_cols=["relative_gain_pct"], int_cols=["queries"])
    return df


def t3b(f1):
    rows = []
    for g in ("under_1M", "over_1M"):
        for b in ("BM25", "DPH"):
            s = f1[(f1.group == g) & (f1.base == b)]
            rows.append({"group": GROUP_NAME[g], "base": b, "collections": len(s),
                         "above_+0.005": int((s.gain > FLOOR).sum()), "within_0.005": int(((s.gain >= -FLOOR) & (s.gain <= FLOOR)).sum()),
                         "below_-0.005": int((s.gain < -FLOOR).sum()),
                         "below_names": ";".join(s[s.gain < -FLOOR].dataset), "base_ndcg10_macro": float(s.base_ndcg10.mean()),
                         "mean_gain": float(s.gain.mean()), "mean_gain_rel_pct": rel(float(s.gain.mean()), float(s.base_ndcg10.mean()))})
    df = pd.DataFrame(rows)
    # cross-check against the post hoc risk profile (same S2 rows against the same base)
    rp = pd.read_csv(f"{SL}/paper_posthoc/a_risk_profile.csv")
    for r in rows:
        g = "under_1M" if r["group"] == "under 1M" else "over_1M"
        x = rp[(rp.group == g) & (rp.method == f"{r['base']}+S2")].iloc[0]
        check(f"T3b below {g} {r['base']}", r["below_-0.005"], x["corpora_below_-0.005"], 0)
        check(f"T3b above {g} {r['base']}", r["above_+0.005"], x["corpora_above_+0.005"], 0)
        check(f"T3b mean {g} {r['base']}", r["mean_gain"], x["mean_gain"])
    write("T3b", df, delta_cols=["mean_gain"], level_cols=["base_ndcg10_macro"], pct_cols=["mean_gain_rel_pct"],
          int_cols=["collections", "above_+0.005", "within_0.005", "below_-0.005"])
    return df


# ---------------------------------------------------------------- T4

def t4():
    h = json.load(open(f"{SL}/paper_posthoc/a_holm.json"))
    ac = pd.read_csv(f"{SL}/paper_posthoc/a_contrasts.csv")
    # per-collection levels on S2's query set (all queries)
    lev = []
    for ds in UNDER + OVER:
        p = pd.read_parquet(f"{SL}/prf_comparison/per_query_{ds}.parquet", columns=["qid", "method"] + T4_METS)
        o = pd.read_parquet(f"{SL}/feature_sidecar/outcomes_{ds}.parquet", columns=["qid", "model", "arm"] + T4_METS)
        for model, nm in (("bm25", "BM25"), ("dph", "DPH")):
            s2 = o[(o.model == model) & (o.arm == "S2")].set_index("qid")[T4_METS]
            qs = s2.index
            lev.append({"dataset": ds, "method": f"{nm}+S2", **s2.mean().to_dict(), "n": len(qs)})
            for x in ("RM3", "Bo1", "BGE-QE", "LLM-Q2E"):
                bx = p[p.method == f"{nm}+{x}"].set_index("qid")[T4_METS].reindex(qs)
                if bx.isna().any().any():
                    raise SystemExit(f"{ds} {nm}+{x}: missing queries of S2's set")
                lev.append({"dataset": ds, "method": f"{nm}+{x}", **bx.mean().to_dict(), "n": len(qs)})
        del p, o
    lev = pd.DataFrame(lev)
    lev["group"] = lev.dataset.map(group_of)
    rows = []
    scope = "all"
    for g in ("under_1M", "over_1M"):
        for nm in ("BM25", "DPH"):
            for x in ("RM3", "Bo1", "BGE-QE", "LLM-Q2E"):
                key = f"{nm}+S2 - {nm}+{x}"
                hv = h[scope][g][key]
                L = lev[lev.group == g]
                cm_s2 = L[L.method == f"{nm}+S2"].set_index("dataset").ndcg10
                cm_x = L[L.method == f"{nm}+{x}"].set_index("dataset").ndcg10
                check(f"T4 macro {scope} {g} {key}", float((cm_s2 - cm_x).mean()), hv["macro"], 1e-9)
                a = ac[(ac.scope == scope) & (ac.group == g) & (ac.contrast == key)].set_index("metric")
                check(f"T4 a_contrasts ndcg10 {scope} {g} {key}", float(a.loc["ndcg10", "macro"]), hv["macro"], 1e-12)
                row = {"group": GROUP_NAME[g], "base": nm, "baseline": f"{nm}+{x}",
                       "collections": int(a.loc["ndcg10", "corpora"]), "queries": int(a.loc["ndcg10", "queries"]),
                       "crve_s2_ndcg10": float(cm_s2.mean()), "baseline_ndcg10": float(cm_x.mean()),
                       "diff_ndcg10": hv["macro"], "ci_low": hv["interval"][0], "ci_high": hv["interval"][1],
                       "interval_level": f"{lvl(hv['level'])} Holm (step {hv['holm_step']} of 8)",
                       "relative_diff_pct": rel(hv["macro"], float(cm_x.mean())), "reading": DIRECTION[hv["direction"]]}
                if row["reading"] != zero_reading(*hv["interval"]):
                    raise SystemExit(f"T4 {scope} {g} {key}: stored direction disagrees with its interval")
                for mt in ("ndcg50", "recall1000"):
                    lo, hi = float(a.loc[mt, "ci95_low"]), float(a.loc[mt, "ci95_high"])
                    bl_mt = float(L[L.method == f"{nm}+{x}"][mt].mean())
                    d_mt = float((L[L.method == f"{nm}+S2"].set_index("dataset")[mt] - L[L.method == f"{nm}+{x}"].set_index("dataset")[mt]).mean())
                    check(f"T4 a_contrasts {mt} {scope} {g} {key}", d_mt, float(a.loc[mt, "macro"]), 1e-9)
                    row.update({f"diff_{mt}": float(a.loc[mt, "macro"]), f"{mt}_ci95_low": lo, f"{mt}_ci95_high": hi,
                                f"{mt}_reading": zero_reading(lo, hi), f"baseline_{mt}": bl_mt,
                                f"diff_{mt}_rel_pct": rel(float(a.loc[mt, "macro"]), bl_mt)})
                rows.append(row)
    df = pd.DataFrame(rows)
    write("T4", df, delta_cols=["diff_ndcg10", "ci_low", "ci_high", "diff_ndcg50", "ndcg50_ci95_low", "ndcg50_ci95_high",
                                "diff_recall1000", "recall1000_ci95_low", "recall1000_ci95_high"],
          level_cols=["crve_s2_ndcg10", "baseline_ndcg10", "baseline_ndcg50", "baseline_recall1000"],
          pct_cols=["relative_diff_pct", "diff_ndcg50_rel_pct", "diff_recall1000_rel_pct"], int_cols=["collections", "queries"])
    return df


def main():
    t3df = t3()
    print(t3df.drop(columns=["chunks"]).round(4).to_string(index=False), flush=True)
    figure1_typicality()
    f1 = f1_data()
    tb = t3b(f1)
    print(tb.round(4).to_string(index=False), flush=True)
    t4df = t4()
    print(t4df[["group", "baseline", "diff_ndcg10", "ci_low", "ci_high", "relative_diff_pct", "reading",
                "diff_ndcg50", "diff_ndcg50_rel_pct", "ndcg50_reading", "diff_recall1000", "diff_recall1000_rel_pct",
                "recall1000_reading"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
