#!/usr/bin/env python3
"""
Appendix A and Supplementary Sections S1 and S2 of the paper: Tables A1, A2 (TA1, TA2), S1 to S4 (TB1 to TB4), S5, S6 (TC1, TC2)
and the per-collection comparison with the best-of-both PRF oracle (FB1_data, Section 6.2 and Supplementary Section S1).
Read-only re-aggregation of stored result files; no retrieval.

  python paper/appendix.py   (from the repository root; or run everything with python paper/make_all.py)

The cross-checks compare recomputed values with the stored summaries and with reference constants: numbers printed in the paper
(the table or section is named in each message).
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GROUP_NAME, NAME, OVER, RES, UNDER, group_of, lvl, rel, write  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lib_direct as RD  # noqa: E402  (holm2)
import lib_full_queries as FQ  # noqa: E402  (contrast)
import lib_prf_oracle as PO  # noqa: E402  (GRID, DEFAULT, cfg_name, pick)

SL = RES
PC_DIR, FS_DIR = f"{SL}/prf_comparison", f"{SL}/feature_sidecar"
PO_DIR, OT_DIR, COST = f"{SL}/prf_oracle", f"{SL}/ours_tuned", f"{SL}/cost"
ALL = UNDER + OVER
GROUPS = (("under_1M", UNDER), ("over_1M", OVER))
MODELS = ("bm25", "dph")
MLAB = {"bm25": "BM25", "dph": "DPH"}
SPLIT_SEEDS = range(30)
TOL = 1e-9
issues = []  # cross-check log, printed at the end


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        issues.append(msg)


def reading(lo, hi):
    """Reading of the Holm rows of Tables S1 and S2 (non-inferiority margin -0.002, set in advance)."""
    return "better" if lo > 0 else "worse" if hi < 0 else "non-inferior" if lo > -0.002 else "unclear"


def zero_reading(lo, hi):
    """Interval against zero (95% rows and ranges)."""
    return "uncertain" if lo <= 0 <= hi else ("above zero" if lo > 0 else "below zero")


# --------------------------------------------------------------------------------------------------------------------
# Appendix A
# --------------------------------------------------------------------------------------------------------------------
# method label in prf_comparison/per_query_<c>.parquet -> column key
PC_METHODS = {"{M} (harness)": "base_harness", "{M} (ours)": "base_ours", "{M}+RM3": "rm3", "{M}+Bo1": "bo1", "{M}+BGE-QE": "bge_qe",
              "{M}+LLM-Q2E": "llm_q2e"}
METHOD_ORDER = ["base_ours", "base_harness", "crve_s2", "rm3", "bo1", "bge_qe", "llm_q2e"]
METHOD_LABEL = {"base_ours": "base (ours; reference of CRVE)", "base_harness": "base (harness; reference of PRF/BGE-QE/LLM-Q2E)",
                "crve_s2": "CRVE (S2)", "rm3": "RM3", "bo1": "Bo1", "bge_qe": "BGE-QE", "llm_q2e": "LLM-Q2E"}
METS4 = ["ndcg10", "ndcg20", "ndcg50", "recall100", "recall1000"]


def per_collection_means():
    """{(dataset, model, method_key): Series of metric means}, one collection at a time."""
    rows = []
    for ds in ALL:
        pq = pd.read_parquet(f"{PC_DIR}/per_query_{ds}.parquet")
        fs = pd.read_parquet(f"{FS_DIR}/outcomes_{ds}.parquet")
        for model in MODELS:
            M = MLAB[model]
            sets = {}
            for lab, key in PC_METHODS.items():
                x = pq[pq.method == lab.format(M=M)]
                check(not x.qid.duplicated().any() and len(x) > 0, f"{ds} {model} {key}: per-query rows unique and present")
                sets[key] = x.set_index("qid")
                rows.append({"dataset": ds, "model": model, "method": key, "queries": len(x), **x[METS4].mean().to_dict()})
            s2 = fs[(fs.model == model) & (fs.arm == "S2")].set_index("qid")
            sb = fs[(fs.model == model) & (fs.arm == "base")].set_index("qid")
            qset = set(sets["base_ours"].index)
            check(set(s2.index) == qset and set(sb.index) == qset, f"{ds} {model}: feature_sidecar/ S2/base query set = comparison set")
            d = (sb.ndcg10 - sets["base_ours"].ndcg10.reindex(sb.index)).abs().max()
            check(d < 1e-9, f"{ds} {model}: S2 base equals '{M} (ours)' per query (max |diff| {d:.2e})")
            rows.append({"dataset": ds, "model": model, "method": "crve_s2", "queries": len(s2), **s2[METS4].mean().to_dict()})
        del pq, fs
    return pd.DataFrame(rows)


def stage_A():
    pcm = per_collection_means()
    pcm["group"] = pcm.dataset.map(group_of)
    # harness vs our base, per collection
    w = pcm.pivot_table(index=["dataset", "model"], columns="method", values="ndcg10")
    gap = (w.base_harness - w.base_ours)
    print("harness - ours base, max |diff| per model:", gap.abs().groupby("model").max().round(4).to_dict(),
          "at", gap.abs().groupby("model").idxmax().to_dict(), flush=True)
    check(gap.abs().max() <= 0.0034 + 5e-5, f"harness base within 0.0034 of our base (max {gap.abs().max():.4f})")

    # ---- TA1: per collection nDCG@10 ----
    cols = []
    rows = []
    for gname, corp in GROUPS:
        for ds in corp:
            r = {"group": GROUP_NAME[gname], "collection": NAME[ds],
                 "queries": int(pcm[(pcm.dataset == ds) & (pcm.method == "base_ours") & (pcm.model == "bm25")].queries.iloc[0])}
            for model in MODELS:
                for mk in METHOD_ORDER:
                    c = f"{MLAB[model]} {METHOD_LABEL[mk]}"
                    r[c] = float(w.loc[(ds, model), mk])
                    if c not in cols:
                        cols.append(c)
            rows.append(r)
        sub = pd.DataFrame(rows[-len(corp):])
        rows.append({"group": GROUP_NAME[gname], "collection": f"Macro {GROUP_NAME[gname]} ({len(corp)} collections)",
                     "queries": int(sub.queries.sum()), **sub[cols].mean().to_dict()})
    ta1 = pd.DataFrame(rows)
    write("TA1", ta1, level_cols=cols, int_cols=["queries"])

    # cross-check against reference constants: rows of Table A1 as printed (Base, Baseh, S2, RM3, Bo1, BGE, LLM over BM25, then DPH)
    rep = {"scifact": "0.691 0.691 0.725 0.679 0.644 0.672 0.695 0.684 0.684 0.722 0.669 0.648 0.664 0.687",
           "quora": "0.769 0.770 0.756 0.751 0.736 0.771 0.767 0.444 0.444 0.423 0.394 0.359 0.420 0.427",
           "hotpotqa": "0.586 0.586 0.588 0.529 0.531 0.567 0.593 0.617 0.617 0.615 0.565 0.553 0.567 0.626",
           "bright_biology": "0.067 0.067 0.071 0.061 0.093 0.099 0.086 0.239 0.239 0.262 0.293 0.274 0.273 0.262"}
    for ds, v in rep.items():
        got = " ".join(f"{w.loc[(ds, m), k]:.3f}" for m in MODELS for k in ("base_ours", "base_harness", "crve_s2", "rm3", "bo1", "bge_qe", "llm_q2e"))
        check(got == v, f"TA1 {ds} equals Table A1: {got}")

    # ---- TA2: macro deeper metrics per method x group x base ----
    rows = []
    for gname, corp in GROUPS:
        for model in MODELS:
            for mk in METHOD_ORDER:
                x = pcm[(pcm.model == model) & (pcm.method == mk) & pcm.dataset.isin(corp)]
                assert len(x) == len(corp)
                rows.append({"group": GROUP_NAME[gname], "base": MLAB[model], "method": METHOD_LABEL[mk], "collections": len(corp),
                             **{f"{m}": float(x[m].mean()) for m in METS4}})
    ta2 = pd.DataFrame(rows).rename(columns={"ndcg10": "nDCG@10", "ndcg20": "nDCG@20", "ndcg50": "nDCG@50", "recall100": "Recall@100",
                                             "recall1000": "Recall@1000"})
    write("TA2", ta2, level_cols=["nDCG@10", "nDCG@20", "nDCG@50", "Recall@100", "Recall@1000"], int_cols=["collections"])
    # cross-check against reference constants: rows of Table A2 as printed (nDCG@10, @20, @50, Recall@100, @1000)
    t = ta2.set_index(["group", "base", "method"])
    for key, v in ((("under 1M", "BM25", "CRVE (S2)"), "0.252 0.263 0.279 0.431 0.685"), (("over 1M", "DPH", "LLM-Q2E"), "0.427 0.444 0.462 0.688 0.823"),
                   (("under 1M", "BM25", METHOD_LABEL["base_harness"]), "0.243 0.254 0.269 0.420 0.678")):
        got = " ".join(f"{t.loc[key, m]:.3f}" for m in ("nDCG@10", "nDCG@20", "nDCG@50", "Recall@100", "Recall@1000"))
        check(got == v, f"TA2 {key} equals Table A2: {got}")
    return w


# --------------------------------------------------------------------------------------------------------------------
# Supplementary Section S1 (tuning on both sides)
# --------------------------------------------------------------------------------------------------------------------
def prf_cv_by_split():
    """Per collection x model x method x seed: mean nDCG@10 of cross-validated PRF (two folds, the grid of lib_prf_oracle), seeds 0-29 and 42.
    Also returns the seed-42 per-query rows for the parity check and the capped query sets."""
    arms = pd.read_parquet(f"{PO_DIR}/per_query_arms.parquet", columns=["dataset", "model", "arm", "qid", "ndcg10"])
    rows, qsets, mism = [], {}, []
    for ds in ALL:
        g = pd.read_parquet(f"{PO_DIR}/grid_{ds}.parquet", columns=["qid", "model", "method", "config", "ndcg10"])
        # the capped set: query_subset(ds) is sorted; every stored arm holds exactly it
        q_ids = sorted(arms[(arms.dataset == ds) & (arms.model == "bm25") & (arms.arm == "rm3_cv")].qid.unique())
        check(set(g.qid) == set(q_ids) and len(q_ids) <= 500, f"{ds}: grid query set = capped set ({len(q_ids)} queries)")
        qsets[ds] = q_ids
        for model in MODELS:
            for method in ("rm3", "bo1"):
                pq = g[(g.model == model) & (g.method == method)]
                for s in list(SPLIT_SEEDS) + [42]:
                    half = set(random.Random(s).sample(q_ids, len(q_ids) // 2))
                    fa, fb = [x for x in q_ids if x in half], [x for x in q_ids if x not in half]
                    ca, cb = PO.pick(pq, fa), PO.pick(pq, fb)
                    cv = pd.concat([pq[(pq.config == cb) & pq.qid.isin(fa)], pq[(pq.config == ca) & pq.qid.isin(fb)]])
                    rows.append({"dataset": ds, "model": model, "method": method, "seed": s, "cv": float(cv.ndcg10.mean())})
                    if s == 42:
                        st = arms[(arms.dataset == ds) & (arms.model == model) & (arms.arm == f"{method}_cv")].set_index("qid").ndcg10
                        d = (cv.set_index("qid").ndcg10.reindex(st.index) - st).abs().max()
                        if not (d < 1e-12 and len(cv) == len(st)):
                            mism.append((ds, model, method, d))
                default = pq[pq.config == PO.cfg_name(PO.DEFAULT[method])].ndcg10.mean()
                rows.append({"dataset": ds, "model": model, "method": method, "seed": "default", "cv": float(default)})
        del g
    check(not mism, f"recomputed PRF CV at seed 42 equals stored rm3_cv/bo1_cv per query on every collection x model x method {mism[:3]}")
    r = pd.DataFrame(rows)
    r["group"] = r.dataset.map(group_of)
    return r, qsets


def macro_level(df, arm, corp, model):
    x = df[(df.model == model) & (df.arm == arm) & df.dataset.isin(corp)]
    return float(x.groupby("dataset").ndcg10.mean().mean())


def tb1_column(df, ours_arm, cvsplit, ours_permean):
    """Rows group x model x comparison for one CRVE variant. df: long per-query frame holding ours_arm and the PRF arms."""
    out = []
    for gname, corp in GROUPS:
        fam, res = {}, {}
        for model in MODELS:
            sub = df[(df.model == model) & df.dataset.isin(corp)]
            for b in ("rm3_oracle", "bo1_oracle", "prf_oracle", "rm3_cv", "bo1_cv"):
                c = FQ.contrast(sub, ours_arm, b, "ndcg10")
                res[(model, b)] = c
                if b in ("rm3_oracle", "bo1_oracle"):
                    fam[f"{model}: ours - {b}"] = c
        h = RD.holm2(fam)
        for model in MODELS:
            for b in ("rm3_oracle", "bo1_oracle", "prf_oracle", "rm3_cv", "bo1_cv"):
                c = res[(model, b)]
                ref = macro_level(df, b, corp, model)
                if b in ("rm3_oracle", "bo1_oracle"):
                    hh = h[f"{model}: ours - {b}"]
                    lo, hi = hh["interval"]
                    itype, level, rd = "Holm (4 per group)", hh["level"], reading(lo, hi)
                else:
                    lo, hi = c["ci95"]
                    itype, level, rd = "95%", 0.95, zero_reading(lo, hi)
                out.append({"group": GROUP_NAME[gname], "base": MLAB[model], "comparison": f"{b} (seed-42 split)" if b.endswith("_cv") else b,
                            "diff": c["macro"], "low": lo, "high": hi, "interval": itype, "level": level, "reading": rd,
                            "reference_ndcg10": ref, "rel_pct": rel(c["macro"], ref)})
            # 30-split cross-validated PRF: mean, min, max of (ours - PRF CV) macro over seeds 0-29
            for method in ("rm3", "bo1"):
                x = cvsplit[(cvsplit.model == model) & (cvsplit.method == method) & cvsplit.dataset.isin(corp)
                            & cvsplit.seed.isin(list(SPLIT_SEEDS))]
                prf_macro = x.groupby("seed").cv.mean()
                ours_macro = float(ours_permean[model].reindex(corp).mean())
                d = ours_macro - prf_macro
                out.append({"group": GROUP_NAME[gname], "base": MLAB[model], "comparison": f"{method}_cv (30 splits, seeds 0-29)",
                            "diff": float(d.mean()), "low": float(d.min()), "high": float(d.max()), "interval": "range over 30 splits (min, max)",
                            "level": np.nan, "reading": "uncertain" if d.min() <= 0 <= d.max() else ("above zero in every split" if d.min() > 0 else "below zero in every split"),
                            "reference_ndcg10": float(prf_macro.mean()), "rel_pct": rel(float(d.mean()), float(prf_macro.mean()))})
    return pd.DataFrame(out)


def stage_TB1(cvsplit, qsets):
    po = pd.read_parquet(f"{PO_DIR}/per_query_arms.parquet", columns=["dataset", "model", "arm", "qid", "ndcg10"])
    ot = pd.read_parquet(f"{OT_DIR}/per_query_arms.parquet", columns=["dataset", "model", "arm", "qid", "ndcg10"])
    prf_arms = ["rm3_oracle", "bo1_oracle", "prf_oracle", "rm3_cv", "bo1_cv", "rm3_default", "bo1_default", "base"]
    # the PRF arms of ours_tuned/ are those of prf_oracle/, on the same capped query sets
    a = po[po.arm.isin(prf_arms)].set_index(["dataset", "model", "arm", "qid"]).ndcg10.sort_index()
    b = ot[ot.arm.isin(prf_arms)].set_index(["dataset", "model", "arm", "qid"]).ndcg10.sort_index()
    check(a.index.equals(b.index) and float((a - b).abs().max()) == 0.0, "ours_tuned/ PRF arms equal prf_oracle/ PRF arms per query")
    bad = [(ds, m, arm) for (ds, m, arm), g in ot.groupby(["dataset", "model", "arm"]) if sorted(g.qid) != qsets[ds]]
    check(not bad, f"every ours_tuned/ arm holds exactly the capped query set {bad[:3]}")
    bad = [(ds, m, arm) for (ds, m, arm), g in po.groupby(["dataset", "model", "arm"]) if sorted(g.qid) != qsets[ds]]
    check(not bad, f"every prf_oracle/ arm holds exactly the capped query set {bad[:3]}")

    # main column: CRVE S2 at the default setting
    s2df = ot[ot.arm.isin(["S2_default"] + prf_arms)]
    s2mean = {m: s2df[(s2df.model == m) & (s2df.arm == "S2_default")].groupby("dataset").ndcg10.mean() for m in MODELS}
    main = tb1_column(s2df, "S2_default", cvsplit, s2mean)
    # cross-checks against reference constants: rows of Table S1 as printed
    t = main.set_index(["group", "base", "comparison"])
    for key, v in ((("under 1M", "BM25", "rm3_oracle"), "+0.0100 +0.0032 +0.0170 98.75% better"),
                   (("under 1M", "BM25", "prf_oracle"), "-0.0074 -0.0132 -0.0015 95% below zero"),
                   (("over 1M", "DPH", "bo1_oracle"), "+0.0142 +0.0060 +0.0226 98.75% better"),
                   (("under 1M", "BM25", "bo1_cv (30 splits, seeds 0-29)"), "+0.0036 -0.0001 +0.0073 uncertain")):
        r = t.loc[key]
        got = f"{r['diff']:+.4f} {r.low:+.4f} {r.high:+.4f} " + (f"{lvl(r.level)} " if np.isfinite(r.level) else "") + r.reading
        check(got == v, f"TB1 {key} equals Table S1: {got}")
    # PRF CV against its default, 30 splits, equals ours_tuned/posthoc_cv_gain_over_default.csv (rm3/bo1 rows)
    ph = pd.read_csv(f"{OT_DIR}/posthoc_cv_gain_over_default.csv")
    dflt = cvsplit[cvsplit.seed == "default"].set_index(["dataset", "model", "method"]).cv
    sp = cvsplit[cvsplit.seed != "default"].copy()
    sp["gain"] = sp.cv.values - dflt.reindex(pd.MultiIndex.from_frame(sp[["dataset", "model", "method"]])).values
    mac = sp.groupby(["group", "model", "method", "seed"]).gain.mean().reset_index()
    for r in ph[ph.arm.isin(["rm3", "bo1"])].itertuples():
        x = mac[(mac.group == r.group) & (mac.model == r.model) & (mac.method == r.arm)]
        s30, s42 = x[x.seed != 42].gain, float(x[x.seed == 42].gain.iloc[0])
        check(abs(s30.mean() - r.mean) < 6e-6 and abs(s30.min() - r.min) < 6e-6 and abs(s30.max() - r.max) < 6e-6 and abs(s42 - r.seed42) < 6e-6,
              f"PRF CV-minus-default {r.group} {r.model} {r.arm}: 30-split {s30.mean():+.5f} seed42 {s42:+.5f} equal posthoc csv")

    write("TB1", main, delta_cols=["diff", "low", "high"], level_cols=["reference_ndcg10"], holm_level_cols=["level"], pct_cols=["rel_pct"])
    return main


def stage_TB2():
    an = json.load(open(f"{OT_DIR}/analysis.json"))
    summ = pd.read_csv(f"{OT_DIR}/summary.csv")
    summ = summ[summ.metric == "ndcg10"].set_index(["group", "model", "contrast"])
    sens = pd.read_csv(f"{OT_DIR}/cv_split_sensitivity.csv").set_index(["group", "model", "contrast"])
    rows = []
    for gname, _ in GROUPS:
        absl = an[gname]
        for model in MODELS:
            lv = absl[f"{model}_absolute_ndcg10"]
            for sc in ("S1", "S2"):
                for p in ("rm3", "bo1", "prf"):
                    r = {"group": GROUP_NAME[gname], "base": MLAB[model], "sidecar": sc, "prf": {"rm3": "RM3", "bo1": "Bo1", "prf": "best of both (oracle only)"}[p]}
                    if p != "prf":
                        h = absl[f"{model}: {sc}_oracle - {p}_oracle"]
                        r.update({"oracle_diff": h["macro"], "oracle_low": h["interval"][0], "oracle_high": h["interval"][1],
                                  "oracle_interval": f"Holm (8 per group), level {lvl(h['level'])}", "oracle_reading": h["reading"],
                                  "oracle_rel_pct": rel(h["macro"], lv[f"{p}_oracle"])})
                        check(h["reading"] == reading(*h["interval"]), f"TB2 {gname} {model} {sc}-{p} reading consistent with interval")
                        s = summ.loc[(gname, model, f"{sc}_cv - {p}_cv")]
                        v = sens.loc[(gname, model, f"{sc}_cv - {p}_cv")]
                        r.update({"cv42_diff": s.macro, "cv42_low": s.ci95_low, "cv42_high": s.ci95_high,
                                  "cv42_reading": zero_reading(s.ci95_low, s.ci95_high),
                                  "cv42_rel_pct": rel(s.macro, lv[f"{p}_cv"]),
                                  "cv30_mean": v["mean"], "cv30_min": v["min"], "cv30_max": v["max"]})
                    else:
                        s = summ.loc[(gname, model, f"{sc}_oracle - prf_oracle")]
                        r.update({"oracle_diff": s.macro, "oracle_low": s.ci95_low, "oracle_high": s.ci95_high, "oracle_interval": "95%",
                                  "oracle_reading": zero_reading(s.ci95_low, s.ci95_high),
                                  "oracle_rel_pct": rel(s.macro, lv["prf_oracle"])})
                    rows.append(r)
    tb2 = pd.DataFrame(rows)
    d = ["oracle_diff", "oracle_low", "oracle_high", "cv42_diff", "cv42_low", "cv42_high", "cv30_mean", "cv30_min", "cv30_max"]
    write("TB2", tb2, delta_cols=d, pct_cols=["oracle_rel_pct", "cv42_rel_pct"])
    # cross-checks against reference constants: values printed in Table S2
    t = tb2.set_index(["group", "base", "sidecar", "prf"])
    for key, exp in ((("under 1M", "BM25", "S2", "Bo1"), ("oracle_diff", 0.0030)), (("over 1M", "DPH", "S1", "RM3"), ("oracle_diff", 0.0018)),
                     (("under 1M", "BM25", "S2", "Bo1"), ("cv30_mean", 0.0043)), (("under 1M", "BM25", "S2", "Bo1"), ("cv30_min", -0.0004)),
                     (("over 1M", "DPH", "S1", "RM3"), ("cv30_min", -0.0014)), (("under 1M", "BM25", "S2", "best of both (oracle only)"), ("oracle_diff", -0.0025)),
                     (("under 1M", "DPH", "S1", "Bo1"), ("cv42_diff", 0.0229))):
        g = t.loc[key, exp[0]]
        check(abs(g - exp[1]) < 6e-5, f"TB2 {key} {exp[0]}: {g:+.4f} (Table S2: {exp[1]:+.4f})")
    rd = tb2[tb2.prf != "best of both (oracle only)"].oracle_reading.value_counts().to_dict()
    check(rd == {"better": 11, "non-inferior": 3, "unclear": 2}, f"TB2 Holm readings {rd} (Table S2: 11 better, 3 non-inferior, 2 unclear)")
    return tb2


def stage_TB3():
    an = json.load(open(f"{OT_DIR}/analysis.json"))
    summ = pd.read_csv(f"{OT_DIR}/summary.csv")
    summ = summ[summ.metric == "ndcg10"].set_index(["group", "model", "contrast"])
    ph = pd.read_csv(f"{OT_DIR}/posthoc_cv_gain_over_default.csv").set_index(["group", "model", "arm"])
    arms = pd.read_parquet(f"{OT_DIR}/per_query_arms.parquet", columns=["dataset", "model", "arm", "qid", "ndcg10"])
    rows = []
    for gname, corp in GROUPS:
        for model in MODELS:
            lv = an[gname][f"{model}_absolute_ndcg10"]
            for a in ("S1", "S2", "rm3", "bo1"):
                ref = lv[f"{a}_default"]
                so = summ.loc[(gname, model, f"{a}_oracle - {a}_default")]
                if a in ("S1", "S2"):
                    sc = summ.loc[(gname, model, f"{a}_cv - {a}_default")]
                    cv, lo, hi = sc.macro, sc.ci95_low, sc.ci95_high
                else:  # not in summary.csv: recomputed from the stored per-query arms with the same contrast function
                    sub = arms[(arms.model == model) & arms.dataset.isin(corp) & arms.arm.isin([f"{a}_cv", f"{a}_default"])]
                    c = FQ.contrast(sub, f"{a}_cv", f"{a}_default", "ndcg10")
                    cv, (lo, hi) = c["macro"], c["ci95"]
                p = ph.loc[(gname, model, a)]
                check(abs(p.seed42 - cv) < 6e-6, f"TB3 {gname} {model} {a}: CV-default seed 42 {cv:+.5f} equals posthoc {p.seed42:+.5f}")
                rows.append({"group": GROUP_NAME[gname], "base": MLAB[model], "method": {"S1": "CRVE S1", "S2": "CRVE S2", "rm3": "RM3", "bo1": "Bo1"}[a],
                             "default_ndcg10": ref, "oracle_minus_default": so.macro, "oracle_low": so.ci95_low, "oracle_high": so.ci95_high,
                             "oracle_rel_pct": rel(so.macro, ref), "cv42_minus_default": cv, "cv42_low": lo, "cv42_high": hi,
                             "cv42_rel_pct": rel(cv, ref), "cv30_mean_minus_default": p["mean"], "cv30_min": p["min"], "cv30_max": p["max"],
                             "cv30_rel_pct": rel(p["mean"], ref)})
    del arms
    tb3 = pd.DataFrame(rows)
    write("TB3", tb3, delta_cols=["oracle_minus_default", "oracle_low", "oracle_high", "cv42_minus_default", "cv42_low", "cv42_high",
                                  "cv30_mean_minus_default", "cv30_min", "cv30_max"], level_cols=["default_ndcg10"],
          pct_cols=["oracle_rel_pct", "cv42_rel_pct", "cv30_rel_pct"])
    t = tb3.set_index(["group", "base", "method"])
    for key, col, exp in ((("under 1M", "BM25", "CRVE S1"), "oracle_minus_default", 0.0077), (("over 1M", "DPH", "RM3"), "oracle_minus_default", 0.0137),
                          (("under 1M", "BM25", "CRVE S1"), "cv42_minus_default", 0.0005), (("over 1M", "BM25", "CRVE S2"), "cv42_minus_default", -0.0009),
                          (("under 1M", "DPH", "CRVE S1"), "cv30_mean_minus_default", 0.0061), (("over 1M", "BM25", "RM3"), "cv30_mean_minus_default", 0.0158)):
        check(abs(t.loc[key, col] - exp) < 6e-5, f"TB3 {key} {col}: {t.loc[key, col]:+.4f} (Table S3: {exp:+.4f})")
    return tb3


def stage_TB4():
    r = pd.read_csv(f"{OT_DIR}/risk_profile.csv")
    order = ["S1_default", "S1_oracle", "S1_cv", "S2_default", "S2_oracle", "S2_cv", "rm3_default", "rm3_oracle", "rm3_cv",
             "bo1_default", "bo1_oracle", "bo1_cv", "prf_oracle"]
    lab = {"S1": "CRVE S1", "S2": "CRVE S2", "rm3": "RM3", "bo1": "Bo1"}
    arms = pd.read_parquet(f"{OT_DIR}/per_query_arms.parquet", columns=["dataset", "model", "arm", "qid", "ndcg10"])
    pc = arms.groupby(["model", "arm", "dataset"]).ndcg10.mean()  # per-collection means on the capped sets
    del arms
    rows = []
    for gname, corp in GROUPS:
        for model in MODELS:
            base_c = pc.loc[(model, "base")].reindex(corp)
            base_level = float(base_c.mean())
            for a in order:
                x = r[(r.group == gname) & (r.model == model) & (r.arm == a)].iloc[0]
                g = pc.loc[(model, a)].reindex(corp) - base_c
                check(abs(float(g.mean()) - x.mean_gain_over_base) < 1e-9 and abs(float(g.min()) - x.worst_gain) < 1e-9
                      and g.idxmin() == x.worst_corpus, f"TB4 {gname} {model} {a}: mean and worst gain recomputed from per_query_arms (base arm)")
                meth, tun = ("PRF best of both", "oracle") if a == "prf_oracle" else (lab[a.split("_")[0]], a.split("_")[1])
                rows.append({"group": GROUP_NAME[gname], "base": MLAB[model], "method": meth,
                             "tuning": {"default": "default", "oracle": "oracle", "cv": "cross-validated (seed-42 split)"}[tun],
                             "collections": int(x.corpora), "below_-0.005": int(x["corpora_below_-0.005"]), "above_+0.005": int(x["corpora_above_+0.005"]),
                             "worst_collection": NAME[x.worst_corpus], "worst_gain": x.worst_gain,
                             "worst_collection_base_ndcg10": float(base_c[x.worst_corpus]),
                             "worst_gain_rel_pct": rel(x.worst_gain, float(base_c[x.worst_corpus])),
                             "base_ndcg10_macro": base_level, "mean_gain_over_base": x.mean_gain_over_base,
                             "mean_gain_over_base_rel_pct": rel(x.mean_gain_over_base, base_level)})
    tb4 = pd.DataFrame(rows)
    write("TB4", tb4, delta_cols=["worst_gain", "mean_gain_over_base"], level_cols=["worst_collection_base_ndcg10", "base_ndcg10_macro"],
          pct_cols=["worst_gain_rel_pct", "mean_gain_over_base_rel_pct"], int_cols=["collections", "below_-0.005", "above_+0.005"])
    # cross-check against reference constants: counts printed in Table S4
    t = tb4.set_index(["group", "base", "method", "tuning"])["below_-0.005"]
    exp = {("under 1M", "BM25", "CRVE S1", "cross-validated (seed-42 split)"): 2, ("over 1M", "DPH", "CRVE S2", "default"): 3,
           ("under 1M", "DPH", "Bo1", "cross-validated (seed-42 split)"): 13, ("under 1M", "BM25", "PRF best of both", "oracle"): 5}
    for k, e in exp.items():
        check(t.loc[k] == e, f"TB4 {k}: {t.loc[k]} below -0.005 (Table S4: {e})")
    check(int(t.xs("oracle", level="tuning").loc[(slice(None), slice(None), ["CRVE S1", "CRVE S2"])].sum()) == 0, "TB4 tuned CRVE oracle lowers no collection")
    return tb4


def stage_FB1():
    """Per collection, CRVE (S2) at its fixed setting minus the best-of-both PRF oracle (capped query sets): the counts of
    Section 6.2 and Supplementary Section S1 and the TREC-COVID difference of Section 6.3 (point estimates)."""
    arms = pd.read_parquet(f"{OT_DIR}/per_query_arms.parquet", columns=["dataset", "model", "arm", "qid", "ndcg10"])
    arms = arms[arms.arm.isin(["S2_default", "prf_oracle"])]
    rows = []
    for ds in ALL:
        for model in MODELS:
            p = arms[(arms.dataset == ds) & (arms.model == model)].pivot(index="qid", columns="arm", values="ndcg10")
            assert not p.isna().any().any()
            d = (p.S2_default - p.prf_oracle).to_numpy()
            rows.append({"group": GROUP_NAME[group_of(ds)], "collection": NAME[ds], "dataset": ds, "base": MLAB[model], "queries": len(d),
                         "s2_default_ndcg10": float(p.S2_default.mean()), "prf_oracle_ndcg10": float(p.prf_oracle.mean()),
                         "diff": float(d.mean()), "rel_pct": rel(float(d.mean()), float(p.prf_oracle.mean()))})
    del arms
    fb = pd.DataFrame(rows)
    # sort within size groups by the BM25 difference
    key = fb[fb.base == "BM25"].set_index("dataset")["diff"]
    fb["_g"] = fb.dataset.map(lambda c: 0 if c in UNDER else 1)
    fb["_k"] = fb.dataset.map(key)
    fb = fb.sort_values(["_g", "_k", "base"], ascending=[True, False, True]).drop(columns=["_g", "_k"]).reset_index(drop=True)
    write("FB1_data", fb.drop(columns=["dataset"]), delta_cols=["diff"], level_cols=["s2_default_ndcg10", "prf_oracle_ndcg10"],
          pct_cols=["rel_pct"], int_cols=["queries"])
    # reference constants: ahead / behind / within +-0.005 (Supplementary Section S1: 10, 9, 6 over BM25 and 9, 10, 6 over DPH)
    for model, exp in (("BM25", (10, 9, 6)), ("DPH", (9, 10, 6))):
        x = fb[fb.base == model]["diff"]
        got = (int((x > 0.005).sum()), int((x < -0.005).sum()), int((x.abs() <= 0.005).sum()))
        check(got == exp, f"FB1 {model}: CRVE (S2) vs best-of-both oracle ahead/behind/within +-0.005 {got} (Supplementary Section S1: {exp})")
    tc = fb[(fb.dataset == "trec_covid") & (fb.base == "BM25")].iloc[0]
    check(f"{tc['diff']:+.3f} {tc.rel_pct:+.1f}" == "-0.010 -1.5",
          f"FB1 TREC-COVID over BM25: {tc['diff']:+.4f} ({tc.rel_pct:+.1f}%) (Section 6.3: -0.010, -1.5%)")
    return fb


# --------------------------------------------------------------------------------------------------------------------
# Supplementary Section S2 (cost)
# --------------------------------------------------------------------------------------------------------------------
COST_NOTE = ("Research implementation (Python, through PyTerrier) set against Terrier's native PRF implementations; most of our overhead "
             "over BM25 is implementation cost (per-call PyTerrier overhead in the sparse lexical channel of Gate 1, Python-side query "
             "encoding) rather than cost inherent to the method.")


def stage_TC1():
    b = pd.read_csv(f"{COST}/build_costs.csv").set_index("dataset")
    rows = []
    for gname, corp in GROUPS:
        for ds in corp:
            x = b.loc[ds]
            lb = ds in OVER
            check((x.ppmi_scope != "full") == lb, f"TC1 {ds}: PPMI scope '{x.ppmi_scope}' consistent with size group")
            rows.append({"group": GROUP_NAME[gname], "collection": NAME[ds], "documents": int(x.documents),
                         "build_s": float(x.build_s_total_min_records), "build_min": float(x.build_s_total_min_records) / 60,
                         "of_which_encoding_s": float(x.gate2_encode_s), "peak_rss_gib": float(x.peak_rss_gib_max),
                         "extra_disk_mb": float(x.extra_disk_mb), "terrier_index_mb": float(x.terrier_index_mb),
                         "extra_disk_pct_of_index": 100 * float(x.extra_disk_share_of_index), "gate2_chunks": int(x.gate2_chunks),
                         "lower_bound": lb,
                         "notes": ("lower bound: PPMI sidecar restricted to tested-query anchors; peak RSS excludes the PPMI step (not measured; "
                                   "the full PPMI sidecar does not fit the RAM budget with the current code)") if lb else "full build"})
    tc1 = pd.DataFrame(rows)
    write("TC1", tc1, pct_cols=["build_s", "build_min", "of_which_encoding_s", "extra_disk_mb", "terrier_index_mb", "extra_disk_pct_of_index"],
          level_cols=["peak_rss_gib"], int_cols=["documents", "gate2_chunks"])
    u, o = tc1[~tc1.lower_bound], tc1[tc1.lower_bound]
    print(f"TC1 under 1M: median build {u.build_s.median():.0f} s ({u.build_s.min():.0f} s to {u.build_min.max():.0f} min), "
          f"median disk {u.extra_disk_mb.median():.0f} MB ({u.extra_disk_mb.min():.0f}-{u.extra_disk_mb.max():.0f}), "
          f"median share {u.extra_disk_pct_of_index.median():.0f}% ({u.extra_disk_pct_of_index.min():.0f}-{u.extra_disk_pct_of_index.max():.0f}%, "
          f">100% on {(u.extra_disk_pct_of_index > 100).sum()}), peak RSS {u.peak_rss_gib.min():.1f}-{u.peak_rss_gib.max():.1f} GiB; "
          f"over 1M: build {o.build_min.min():.0f}-{o.build_min.max():.0f} min, disk {o.extra_disk_mb.min():.0f}-{o.extra_disk_mb.max():.0f} MB, "
          f"share {o.extra_disk_pct_of_index.min():.0f}-{o.extra_disk_pct_of_index.max():.0f}%, peak RSS {o.peak_rss_gib.min():.1f}-{o.peak_rss_gib.max():.1f} GiB", flush=True)
    # reference constants: rows of Table S5 as printed (build min, encoding s, peak RSS GiB, extra disk MB, index MB, disk / index %, chunks)
    t = tc1.set_index("collection")
    for c, v in (("SciFact", "0.8 7.2 2.378 61.1 25.5 239.4 18942"), ("Touché-2020", "35.9 57.1 7.628 405.2 685.1 59.2 183630"),
                 ("NQ", "62.4 59.8 2.476 324.7 2304.9 14.1 193789"), ("Climate-FEVER", "133.6 62.2 2.515 349.3 5555.7 6.3 197099")):
        r = t.loc[c]
        got = (f"{r.build_min:.1f} {r.of_which_encoding_s:.1f} {r.peak_rss_gib:.3f} {r.extra_disk_mb:.1f} {r.terrier_index_mb:.1f} "
               f"{r.extra_disk_pct_of_index:.1f} {int(r.gate2_chunks)}")
        check(got == v, f"TC1 {c} equals Table S5: {got}")
    return tc1


def stage_TC2():
    s = pd.read_csv(f"{COST}/latency_summary.csv")
    five = ["fiqa", "quora", "bright_biology", "nq", "hotpotqa"]
    lab = {"bm25": "BM25", "bm25_rm3": "BM25+RM3", "bm25_bo1": "BM25+Bo1", "bm25_bge_qe": "BM25+BGE-QE (query encoding on GPU, included)",
           "bm25_llm_q2e": "BM25+LLM-Q2E", "bm25_ours": "BM25+CRVE (query encoding on CPU; measured configuration)"}
    stages = ["total", "gate1", "encode", "gate2", "retrieval"]
    cols = [f"{st}_ms_{q}" for st in stages for q in ("p50", "p90")]
    notes = {"bm25_bge_qe": "BGE-QE's query encoding runs on GPU as in its pipeline",
             "bm25_llm_q2e": "qwen3.5:4b through Ollama, fresh generation on every timed query; model reloaded every 20 queries",
             "bm25_ours": COST_NOTE}

    def q90(x):
        return float(np.percentile(x.dropna(), 90)) if x.notna().any() else np.nan
    # the GPU rows come from a separate timing session (latency_<c>_opt.parquet); its own CPU run and the HotpotQA comparison
    part_a = {ds: float(pd.read_parquet(f"{COST}/latency_{ds}_opt.parquet").query("method == 'ours_A_cpu'").total_ms.median()) for ds in five}
    gpu_hq = float(pd.read_parquet(f"{COST}/latency_hotpotqa_opt.parquet").query("method == 'ours_B_gpu'").total_ms.median())
    cpu_hq = float(s[(s.method == "bm25_ours") & (s.dataset == "hotpotqa")].total_ms_p50.iloc[0])
    check(round(min(part_a.values())) == 78 and round(max(part_a.values())) == 103 and round(gpu_hq) == 108 and gpu_hq > cpu_hq,
          f"TC2 separate timing session: CPU run {min(part_a.values()):.1f}-{max(part_a.values()):.1f} ms (Table S6, note c: 78 to 103); HotpotQA GPU "
          f"{gpu_hq:.1f} ms > CPU row {cpu_hq:.1f} ms")
    GPU_NOTE = (f"From a separate timing session (latency_<collection>_opt.parquet), not the session "
                f"of the other rows; that session's own CPU run gives {min(part_a.values()):.0f}-{max(part_a.values()):.0f} ms; on HotpotQA "
                f"the GPU row ({gpu_hq:.0f} ms) is slower than the CPU row ({cpu_hq:.0f} ms) through retrieval variation"
                f". Same top-5 selections as the CPU configuration on every query. ")
    rows = []
    for m in lab:
        for ds in five:
            x = s[(s.method == m) & (s.dataset == ds)].iloc[0]
            rows.append({"method": lab[m], "collection": NAME[ds], "queries": np.nan, **{c: x[c] for c in cols}, "notes": notes.get(m, "")})
        if m == "bm25_ours":
            for ds in five:
                o = pd.read_parquet(f"{COST}/latency_{ds}_opt.parquet")
                o = o[o.method == "ours_B_gpu"]
                r = {"method": "BM25+CRVE (query encoding on GPU; separate timing session)", "collection": NAME[ds], "queries": len(o)}
                for st in stages:
                    r[f"{st}_ms_p50"], r[f"{st}_ms_p90"] = float(o[f"{st}_ms"].median()), q90(o[f"{st}_ms"])
                r["notes"] = GPU_NOTE + COST_NOTE
                rows.append(r)
    # queries per collection for the summary rows, and a recomputation check of the CRVE CPU row
    for ds in five:
        bdf = pd.read_parquet(f"{COST}/latency_{ds}_base.parquet")
        n = int(bdf[bdf.method == "bm25"].shape[0])
        for r in rows:
            if r["collection"] == NAME[ds] and pd.isna(r["queries"]):
                r["queries"] = n
        o = bdf[bdf.method == "bm25_ours"]
        x = s[(s.method == "bm25_ours") & (s.dataset == ds)].iloc[0]
        check(all(abs(round(float(o[f"{st}_ms"].median()), 1) - x[f"{st}_ms_p50"]) < 0.051 and abs(round(q90(o[f"{st}_ms"]), 1) - x[f"{st}_ms_p90"]) < 0.051
                  for st in stages), f"TC2 {ds}: CRVE CPU medians/p90 recomputed from parquet equal latency_summary.csv")
    tc2 = pd.DataFrame(rows)
    write("TC2", tc2, pct_cols=cols, int_cols=["queries"])
    g = tc2[tc2.method.str.contains("GPU; separate timing session")]
    print(f"TC2 CRVE GPU total median {g.total_ms_p50.min():.0f}-{g.total_ms_p50.max():.0f} ms; encode {g.encode_ms_p50.min():.1f}-{g.encode_ms_p50.max():.1f} ms", flush=True)
    got = " ".join(f"{x:.1f}" for x in g.total_ms_p50)
    check(got == "66.2 52.4 67.4 73.8 108.3", f"TC2 CRVE encoding on GPU, total median per collection {got} (Table S6: 66.2, 52.4, 67.4, 73.8, 108.3)")
    return tc2


def main():
    if not os.path.isdir(RES):
        raise SystemExit(f"stored results not found: {RES}")
    stage_A()
    cvsplit, qsets = prf_cv_by_split()
    stage_TB1(cvsplit, qsets)
    stage_TB2()
    stage_TB3()
    stage_TB4()
    stage_FB1()
    stage_TC1()
    stage_TC2()
    print("\nCROSS-CHECK FAILURES:" if issues else "\nall cross-checks passed", flush=True)
    for i in issues:
        print("  " + i)
    if issues:
        raise SystemExit(f"{len(issues)} cross-check(s) failed")


if __name__ == "__main__":
    main()
