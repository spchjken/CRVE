#!/usr/bin/env python3
"""
Paper Sections 5.2 (robustness) and 5.3 (ablation and sensitivity): T5, T5b, T6, F2, F3, T7, T8, pool size and per-term weighting.
Read-only re-aggregation of stored result files; no retrieval.

  python paper/s52_53.py     (from the repository root; or run everything with python paper/make_all.py)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import NAME, OUT, OVER, RES, UNDER, GROUP_NAME, lvl, rel, write

import numpy as np
import pandas as pd

import lib_full_queries as FQ  # contrast(): corpus-stratified bootstrap, 10,000 replicates, seed 42 (as the sources)

SL = RES
GROUPS = (("under_1M", UNDER), ("over_1M", OVER))
CHECKS = []  # (item, what, ours, reference, ok)


def check(item, what, ours, ref, tol=5e-5):
    ok = (ours == ref) if isinstance(ref, str) else bool(abs(float(ours) - float(ref)) <= tol)
    CHECKS.append({"item": item, "check": what, "computed": ours, "reference": ref, "ok": ok})


def macro_level(df, value="ndcg10"):
    """Equal collection weights: mean over collections of each collection's query mean."""
    return float(df.groupby("dataset")[value].mean().mean())


def ri_per(d):
    return ((d > 0).sum() - (d < 0).sum()) / len(d)


# --------------------------------------------------------------------------------------------------------------- T5
def t5():
    risk = pd.read_csv(f"{SL}/paper_posthoc/a_risk_profile.csv")
    ri = pd.read_csv(f"{SL}/paper_posthoc/a_robustness_index.csv")
    m = risk.merge(ri, on=["group", "method", "base", "corpora"], how="inner", validate="1:1")
    m = m[m.method.str.split("+").str[1].isin(["S2", "RM3", "Bo1", "BGE-QE", "LLM-Q2E"])]  # the rows of Table 4
    # base levels (nDCG@10, macro) from the same per-query files paper_posthoc used for the bases
    pq = pd.concat([pd.read_parquet(f"{SL}/prf_comparison/per_query_{ds}.parquet", columns=["dataset", "qid", "method", "ndcg10"])
                    for ds in UNDER + OVER], ignore_index=True)
    pq = pq[pq.method.isin(["BM25 (ours)", "BM25 (harness)", "DPH (ours)", "DPH (harness)"])]
    lev = {(g, b): macro_level(pq[(pq.method == b) & pq.dataset.isin(c)]) for g, c in GROUPS for b in pq.method.unique()}
    lev_c = pq.groupby(["dataset", "method"]).ndcg10.mean().to_dict()  # (dataset, base run) -> that collection's base level
    mdisp = {"S2": "CRVE", "RM3": "RM3", "Bo1": "Bo1", "BGE-QE": "BGE-QE", "LLM-Q2E": "LLM-Q2E"}
    rows = []
    for r in m.to_dict("records"):
        model, arm = r["method"].split("+", 1)
        variant = "S2 sidecar" if arm == "S2" else "baseline, fixed settings"
        base_level = lev[(r["group"], r["base"])]
        worst_base = lev_c[(r["worst_corpus"], r["base"])]
        rows.append({"method": mdisp[arm], "variant": variant, "base": model, "base_run": r["base"], "group": GROUP_NAME[r["group"]],
                     "query_set": "all queries", "collections": r["corpora"], "base_ndcg10_macro": base_level,
                     "mean_gain": r["mean_gain"], "mean_gain_rel_pct": rel(r["mean_gain"], base_level),
                     "collections_above_+0.005": r["corpora_above_+0.005"], "collections_below_-0.005": r["corpora_below_-0.005"],
                     "worst_collection": NAME[r["worst_corpus"]], "worst_gain": r["worst_gain"],
                     "worst_collection_base_ndcg10": worst_base, "worst_gain_rel_pct": rel(r["worst_gain"], worst_base),
                     "ri_mean_per_collection": r["ri_macro"], "ri_pooled": r["ri_pooled"]})
    # tuned PRF on the capped query sets of prf_oracle/ (the counts of Section 5.2), re-aggregated as above
    pa = pd.read_parquet(f"{SL}/prf_oracle/per_query_arms.parquet", columns=["dataset", "qid", "model", "arm", "ndcg10"])
    base = pa[pa.arm == "base"][["dataset", "qid", "model", "ndcg10"]].rename(columns={"ndcg10": "b"})
    # the base arm of prf_oracle/ is our base ("BM25 (ours)" / "DPH (ours)" of prf_comparison/) on the capped queries
    for model in ("bm25", "dph"):
        ours = pq[pq.method == f"{model.upper()} (ours)"].set_index(["dataset", "qid"]).ndcg10
        mb = base[base.model == model].set_index(["dataset", "qid"]).b
        hit = ours.reindex(mb.index)
        check("T5", f"prf_oracle/ base arm ({model}) equals '{model.upper()} (ours)' per query on the capped sets (max |diff|)",
              float((hit - mb).abs().max()) if not hit.isna().any() else float("inf"), 0.0, 1e-12)
    x = pa[pa.arm != "base"].merge(base, on=["dataset", "qid", "model"], validate="m:1")
    x["d"] = x.ndcg10 - x.b
    ref_rp = pd.read_csv(f"{SL}/prf_oracle/risk_profile.csv")
    tuned = {"prf_oracle": ("RM3/Bo1 best-of-both oracle", "tuned PRF, oracle per collection"),
             "rm3_cv": ("RM3", "tuned PRF, cross-validated"), "bo1_cv": ("Bo1", "tuned PRF, cross-validated")}
    for g, corp in GROUPS:
        for model in ("bm25", "dph"):
            mb = base[(base.model == model) & base.dataset.isin(corp)].rename(columns={"b": "ndcg10"})
            base_level = macro_level(mb)
            base_c = mb.groupby("dataset").ndcg10.mean()
            for arm, (meth, var) in tuned.items():
                s = x[(x.model == model) & (x.arm == arm) & x.dataset.isin(corp)]
                gain = s.groupby("dataset").ndcg10.mean() - s.groupby("dataset").b.mean()
                per = s.groupby("dataset").d.apply(ri_per)
                rows.append({"method": meth, "variant": var, "base": model.upper(), "base_run": f"{model.upper()} (ours)",
                             "group": GROUP_NAME[g], "query_set": "capped <=500 queries per collection (prf_oracle/)",
                             "collections": int(len(gain)), "base_ndcg10_macro": base_level, "mean_gain": float(gain.mean()),
                             "mean_gain_rel_pct": rel(float(gain.mean()), base_level),
                             "collections_above_+0.005": int((gain > 0.005).sum()), "collections_below_-0.005": int((gain < -0.005).sum()),
                             "worst_collection": NAME[gain.idxmin()], "worst_gain": float(gain.min()),
                             "worst_collection_base_ndcg10": float(base_c[gain.idxmin()]),
                             "worst_gain_rel_pct": rel(float(gain.min()), float(base_c[gain.idxmin()])),
                             "ri_mean_per_collection": float(per.mean()), "ri_pooled": float(ri_per(s.d))})
                rr = ref_rp[(ref_rp.group == g) & (ref_rp.model == model) & (ref_rp.arm == arm)].iloc[0]
                check("T5", f"{g} {model} {arm} mean gain vs prf_oracle/risk_profile.csv", float(gain.mean()), rr.mean_gain_over_base, 1e-9)
                check("T5", f"{g} {model} {arm} below/above/worst", f"{int((gain < -0.005).sum())}/{int((gain > 0.005).sum())}/{gain.idxmin()}",
                      f"{rr['corpora_below_-0.005']}/{rr['corpora_above_+0.005']}/{rr.worst_corpus}")
    df = pd.DataFrame(rows)
    df["notes"] = np.where(df.query_set.str.startswith("capped"),
                           "capped query sets of prf_oracle/; against our base (the base arm of prf_oracle/ = "
                           + df.base + " (ours))",
                           np.where(df.method == "CRVE", "all queries; against our base",
                                    "all queries; baseline against the harness base"))
    write("T5", df, delta_cols=["mean_gain", "worst_gain"],
          level_cols=["base_ndcg10_macro", "worst_collection_base_ndcg10", "ri_mean_per_collection", "ri_pooled"],
          pct_cols=["mean_gain_rel_pct", "worst_gain_rel_pct"], int_cols=["collections", "collections_above_+0.005", "collections_below_-0.005"])
    return df


# -------------------------------------------------------------------------------------------------------------- T5b
def t5b():
    """Quora by BM25 level (Section 6.2): CRVE (S2) over our BM25, per stratum of the BM25 nDCG@10 of each query."""
    c = pd.read_csv(f"{SL}/added_term_profile/ceiling_by_bm25_level.csv")
    c = c.rename(columns={"stratum": "bm25_level", "queries": "queries", "share_of_queries": "share_of_queries",
                          "changed": "share_changed", "improved": "share_improved", "degraded": "share_degraded",
                          "mean_delta": "mean_delta_ndcg10", "contribution_to_overall_mean": "contribution_to_overall_mean",
                          "base_mean": "bm25_ndcg10_all_queries", "arm_mean": "method_ndcg10_all_queries"})
    c = c[c["method"] == "CRVE S2"].copy()
    c.insert(0, "collection", "Quora")
    c["method"] = "CRVE (S2)"
    c["share_of_queries_pct"] = 100 * c.share_of_queries
    for k in ("changed", "improved", "degraded"):
        c[f"share_{k}_pct"] = 100 * c[f"share_{k}"]
    # overall change relative to BM25's level on these queries
    c["overall_rel_pct"] = [rel(a - b, b) for a, b in zip(c.method_ndcg10_all_queries, c.bm25_ndcg10_all_queries)]
    c = c.drop(columns=["share_of_queries", "share_changed", "share_improved", "share_degraded"]).reset_index(drop=True)
    c["notes"] = "post hoc; strata by the nDCG@10 of our BM25 on each query; base levels printed with 4 decimals"
    x = c.set_index("bm25_level")
    check("T5b", "share of Quora queries with BM25 nDCG@10 = 1 (Section 6.2: 60%)", round(float(x.loc["BM25 = 1", "share_of_queries_pct"])), 60, 0)
    share_loss = float(x.loc["BM25 = 1", "contribution_to_overall_mean"]) / float(x.loc["all", "mean_delta_ndcg10"])
    check("T5b", "share of CRVE's net loss on Quora from BM25 = 1 queries (Section 6.2: about four fifths)", round(5 * share_loss), 4, 0)
    write("T5b", c, delta_cols=["mean_delta_ndcg10", "contribution_to_overall_mean"],
          level4_cols=["bm25_ndcg10_all_queries", "method_ndcg10_all_queries"],
          pct_cols=["share_of_queries_pct", "share_changed_pct", "share_improved_pct", "share_degraded_pct", "overall_rel_pct"],
          int_cols=["queries"])
    return c


# --------------------------------------------------------------------------------------------------------------- T6
def t6():
    """Table A5: Gate 2 against the order of a five-channel Gate 1 (randomly drawn chunks, BM25), 18 collections under 1M."""
    R = f"{SL}/recheck"
    parts = []
    for d in (f"{R}/unseen_queries", f"{R}/new_corpora"):
        q = pd.read_parquet(f"{d}/query_outcomes_F.parquet", columns=["dataset", "qid", "ranking", "cap", "ndcg10", "base_ndcg10"])
        keep = pd.read_csv(f"{d}/queries.csv", dtype={"qid": str})
        keep = keep[keep.role == "unseen"][["dataset", "qid"]]  # the evaluated queries of these collections
        parts.append(q.merge(keep, on=["dataset", "qid"]))
    q = pd.concat(parts, ignore_index=True)
    q = q[q.ranking.isin(["r20", "proposer"]) & q.cap.isin([5, 10])]
    paired_src = pd.read_csv(f"{R}/gate1_order_paired.csv")
    own_src = pd.read_csv(f"{R}/gate1_order_18_corpora.csv")
    unc = lambda ci: "uncertain" if ci[0] <= 0 <= ci[1] else ("above zero" if ci[0] > 0 else "below zero")
    arm_name = {"r20": "full method", "proposer": "Gate 1 order"}
    rows = []
    for k in (5, 10):
        s = q[q.cap == k]
        own = {}
        for rk in ("r20", "proposer"):  # full method (Gate 2 over the random sidecar) and Gate 1 order alone
            t = s[s.ranking == rk]
            df = pd.concat([t[["dataset", "qid"]].assign(arm="x", ndcg10=t.ndcg10.values),
                            t[["dataset", "qid"]].assign(arm="base", ndcg10=t.base_ndcg10.values)])
            own[rk] = FQ.contrast(df, "x", "base", "ndcg10")
            own[rk]["queries"] = int(len(t))
            o = own_src[(own_src.when == "after") & (own_src.ranking == rk) & (own_src.K == k)].iloc[0]
            check("T6", f"K={k} {arm_name[rk]} gain vs gate1_order_18_corpora.csv", own[rk]["macro"], o.macro_gain, 1e-12)
            check("T6", f"K={k} {arm_name[rk]} gain ci low", own[rk]["ci95"][0], o.ci95_low, 1e-12)
            check("T6", f"K={k} {arm_name[rk]} gain ci high", own[rk]["ci95"][1], o.ci95_high, 1e-12)
        bl = macro_level(s[s.ranking == "r20"].rename(columns={"ndcg10": "_x", "base_ndcg10": "ndcg10"}))
        c = FQ.contrast(s[["dataset", "qid", "ranking", "ndcg10"]].rename(columns={"ranking": "arm"}), "r20", "proposer", "ndcg10")
        ps = paired_src[(paired_src.when == "after") & (paired_src.scope == "all_18") & (paired_src.K == k)].iloc[0]
        check("T6", f"K={k} paired Gate 2 contribution vs gate1_order_paired.csv", c["macro"], ps.r20_minus_gate1_order, 1e-12)
        check("T6", f"K={k} paired interval low", c["ci95"][0], ps.ci95_low, 1e-12)
        full, g1 = own["r20"], own["proposer"]
        rows.append({"K": k, "collections": c["n_corpora"], "queries": full["queries"], "bm25_ndcg10_macro": bl,
                     "full_method_gain": full["macro"], "full_method_gain_rel_pct": rel(full["macro"], bl),
                     "gate1_order_gain": g1["macro"], "gate1_order_ci95_low": g1["ci95"][0], "gate1_order_ci95_high": g1["ci95"][1],
                     "gate1_order_gain_rel_pct": rel(g1["macro"], bl), "gate1_order_reading": unc(g1["ci95"]),
                     "gate2_contribution": c["macro"], "gate2_contribution_ci95_low": c["ci95"][0], "gate2_contribution_ci95_high": c["ci95"][1],
                     "gate2_contribution_reading": unc(c["ci95"]),
                     "notes": "18 collections under 1M, all their evaluated queries; randomly drawn chunks (20 per term, seed 42); "
                              "five-channel Gate 1; w = 0.20; BM25; 95% intervals"})
    df = pd.DataFrame(rows)
    write("T6", df, delta_cols=["full_method_gain", "gate1_order_gain", "gate1_order_ci95_low", "gate1_order_ci95_high", "gate2_contribution",
                                "gate2_contribution_ci95_low", "gate2_contribution_ci95_high"],
          level_cols=["bm25_ndcg10_macro"], pct_cols=["full_method_gain_rel_pct", "gate1_order_gain_rel_pct"],
          int_cols=["K", "collections", "queries"])
    return df


# --------------------------------------------------------------------------------------------------------------- F2
def f2():
    """Figure 4a: Gate 1 depth L, collections under 1M, S2 alone and the per-query mean of S1 and S2."""
    D = f"{SL}/gate1_depth"
    rows = []
    s = pd.read_csv(f"{D}/summary.csv", usecols=["variant", "model", "L", "contrast", "metric", "macro", "ci95_low", "ci95_high"])
    s = s[(s.metric == "ndcg10") & s.contrast.str.endswith("- base") & s.variant.isin(["S2", "mean"])]
    o = pd.concat([pd.read_parquet(f"{D}/outcomes_{ds}.parquet", columns=["dataset", "qid", "model", "L", "ndcg10"]) for ds in UNDER])
    o = o[o.L == 0]
    a = json.load(open(f"{D}/analysis.json"))
    for model in ("bm25", "dph"):
        bl = macro_level(o[o.model == model])
        for r in s[s.model == model].itertuples(index=False):
            rows.append({"group": GROUP_NAME["under_1M"], "base": model.upper(), "variant": {"mean": "mean of S1 and S2 (per query)"}.get(r.variant, r.variant),
                         "L": int(r.L), "gain_ndcg10": r.macro, "ci95_low": r.ci95_low, "ci95_high": r.ci95_high,
                         "base_ndcg10_macro": bl, "gain_rel_pct": rel(r.macro, bl)})
            if r.variant == "mean" and model == "bm25":
                check("F2", f"under_1M BM25 L={r.L} vs analysis.json", r.macro, a["curve_bm25_ndcg10"][str(r.L)]["gain"], 1e-12)
    df = pd.DataFrame(rows).sort_values(["base", "variant", "L"]).reset_index(drop=True)
    mean_curve = df.variant == "mean of S1 and S2 (per query)"
    df["notes"] = "S1/S2 sidecars, two-channel Gate 1, K = 5, w = 0.20; all queries; intervals 95%, unadjusted"
    write("F2_data", df, delta_cols=["gain_ndcg10", "ci95_low", "ci95_high"], level_cols=["base_ndcg10_macro"], pct_cols=["gain_rel_pct"], int_cols=["L"])
    # reference constants: the values printed in Section 5.3 (the mean of S1 and S2 over BM25: +0.0024 at L = 10, +0.0093 at L = 200)
    for L, v in ((10, 0.0024), (200, 0.0093)):
        x = df[(df.base == "BM25") & mean_curve & (df.L == L)].gain_ndcg10.iloc[0]
        check("F2", f"under 1M BM25 L={L} vs reference constant (Section 5.3)", round(x, 4), v, 1e-9)
    return df


# --------------------------------------------------------------------------------------------------------------- F3
def f3():
    """Figure 4b: chunk grid at K = 5 over BM25 (18 collections under 1M; before the evaluation corrections)."""
    s = pd.read_csv(f"{SL}/chunk_grid/surface.csv", usecols=["scope", "ranking", "length", "chunks", "K", "macro_delta"])
    s = s[(s.scope == "all") & (s.K == 5)].copy()
    qo = pd.concat([pd.read_parquet(f"{SL}/chunk_grid/query_outcomes_{p}.parquet", columns=["dataset", "qid", "ranking", "cap", "base_ndcg10"])
                    for p in ("dev", "new")])
    b = qo[(qo.ranking == "confirmed_r20") & (qo.cap == 5)].drop_duplicates(["dataset", "qid"]).rename(columns={"base_ndcg10": "ndcg10"})
    bl = macro_level(b)
    df = pd.DataFrame({"K": s.K.astype(int), "chunk_length_tokens": s.length, "chunks_per_term": s.chunks,
                       "sidecar": np.where(s.ranking == "confirmed_r20", "CRVE setting (100 tokens, 20 chunks, own sample)", "grid sample subset"),
                       "gain_ndcg10_over_bm25": s.macro_delta, "bm25_ndcg10_macro": bl})
    df.loc[df.sidecar.str.startswith("CRVE setting"), ["chunk_length_tokens", "chunks_per_term"]] = [100.0, 20.0]
    df["gain_rel_pct"] = [rel(v, bl) for v in df.gain_ndcg10_over_bm25]
    df["notes"] = ("computed before the evaluation corrections; 18 collections under 1M, 1,984 queries; "
                   "randomly drawn chunks (seed 42), five-channel Gate 1, w = 0.20; point estimates")
    df = df.sort_values(["K", "sidecar", "chunk_length_tokens", "chunks_per_term"]).reset_index(drop=True)
    write("F3_data", df, delta_cols=["gain_ndcg10_over_bm25"], level_cols=["bm25_ndcg10_macro"], pct_cols=["gain_rel_pct"],
          int_cols=["K", "chunk_length_tokens", "chunks_per_term"])
    return df


# --------------------------------------------------------------------------------------------------------------- T7
def t7():
    import lib_feature_sidecar as FS
    FD, SD = f"{SL}/feature_sidecar", f"{SL}/s4"
    holm_s4 = json.load(open(f"{SD}/primary_holm.json"))
    rows = []
    for g, corp in GROUPS:
        sfx = "" if g == "under_1M" else "_over_1M"
        an = json.load(open(f"{FD}/analysis{sfx}.json"))
        fsum = pd.read_csv(f"{FD}/summary{sfx}.csv", usecols=["model", "contrast", "metric", "macro", "ci95_low", "ci95_high"])
        ssum = pd.read_csv(f"{SD}/summary{sfx}.csv", usecols=["model", "contrast", "metric", "macro", "ci95_low", "ci95_high"])
        summ = pd.concat([fsum, ssum[ssum.contrast.str.startswith("S4")]])
        summ = summ[summ.metric == "ndcg10"]
        # base level and the six draws' gains per model (stored outcomes)
        lev, draws = {}, {}
        for model in ("bm25", "dph"):
            lv, dr = [], []
            for ds in corp:
                o = pd.read_parquet(f"{FD}/outcomes_{ds}.parquet", columns=["qid", "model", "arm", "ndcg10"])
                lv.append(o[(o.model == model) & (o.arm == "base")].ndcg10.mean())
                ref = FS.reference(ds)
                base = ref[(ref.arm == "base") & (ref.model == model)].set_index("qid").ndcg10
                for (arm, seed), x in ref[(ref.model == model) & ref.arm.isin(["both", "seed42"])].groupby(["arm", "seed"]):
                    dr.append({"dataset": ds, "draw": int(seed), "gain": float((x.set_index("qid").ndcg10 - base).mean())})
            lev[model] = float(np.mean(lv))
            draws[model] = pd.DataFrame(dr).groupby("draw").gain.mean()
            if draws[model].size != 6:
                raise SystemExit(f"{g} {model}: expected six draws")
        bg = an["bm25_gain_over_base"]
        check("T7", f"{g} BM25 draw min recomputed vs analysis{sfx}.json", float(draws["bm25"].min()), bg["draw_min"], 1e-12)
        check("T7", f"{g} BM25 draw max recomputed vs analysis{sfx}.json", float(draws["bm25"].max()), bg["draw_max"], 1e-12)
        for model in ("bm25", "dph"):
            dm_gain = summ[(summ.model == model) & (summ.contrast == "draw_mean - base")].macro.iloc[0]
            dm_level = lev[model] + dm_gain
            for rule in ("S2", "S1", "S4"):
                d = summ[(summ.model == model) & (summ.contrast == f"{rule} - draw_mean")].iloc[0]
                gb = summ[(summ.model == model) & (summ.contrast == f"{rule} - base")].iloc[0]
                if model == "bm25":
                    h = holm_s4[g] if rule == "S4" else an[f"{rule} - draw_mean"]
                    lo, hi, level = h["interval"][0], h["interval"][1], h["level"]
                    fam = ("Holm two-sided over S4 under and over 1M" if rule == "S4" else
                           "Holm two-sided over S1, S2 and a third rule combining them (not shown)" if g == "under_1M"
                           else "Holm two-sided over S1, S2")
                    check("T7", f"{g} BM25 {rule} - draw mean: Holm macro vs 95% summary macro", h["macro"], d.macro, 1e-12)
                    itype = f"{fam}, level {lvl(level)}"
                    # margin set in advance: better / worse / non-inferior (lower bound > -0.002) / unclear
                    reading = "better" if lo > 0 else "worse" if hi < 0 else "non-inferior" if lo > -0.002 else "unclear"
                    check("T7", f"{g} BM25 {rule} reading vs stored", reading, h["reading"])
                    note = ("Holm, non-inferiority margin set in advance; reading: better = lower bound > 0, worse = upper bound < 0, "
                            "non-inferior = lower bound > -0.002, else unclear")
                else:
                    lo, hi, level, itype = d.ci95_low, d.ci95_high, 0.95, "95%, unadjusted"
                    reading = "above zero" if lo > 0 else "below zero" if hi < 0 else "uncertain"
                    note = "95%, unadjusted; reading: above zero / below zero / uncertain (interval crosses zero)"
                rows.append({"base": model.upper(), "group": GROUP_NAME[g], "rule": rule, "contrast": f"{rule} minus six-draw mean",
                             "difference_ndcg10": d.macro, "interval_low": lo, "interval_high": hi, "interval": itype, "reading": reading,
                             "draw_mean_ndcg10_level": dm_level, "difference_rel_pct": rel(d.macro, dm_level),
                             "rule_gain_over_base": gb.macro, "rule_gain_rel_pct": rel(gb.macro, lev[model]), "base_ndcg10_macro": lev[model],
                             "draw_mean_gain_over_base": dm_gain, "six_draws_gain_min": float(draws[model].min()),
                             "six_draws_gain_max": float(draws[model].max()),
                             "notes": f"all queries; two-channel Gate 1, K = 5, w = 0.20; {note}"})
    df = pd.DataFrame(rows)
    write("T7", df, delta_cols=["difference_ndcg10", "interval_low", "interval_high", "rule_gain_over_base",
                                "draw_mean_gain_over_base", "six_draws_gain_min", "six_draws_gain_max"],
          level_cols=["draw_mean_ndcg10_level", "base_ndcg10_macro"], pct_cols=["difference_rel_pct", "rule_gain_rel_pct"])
    # reference constants: the values printed in Table 6 (rounded to 4 decimals)
    rep = {("BM25", "under 1M", "S1"): 0.0016, ("BM25", "under 1M", "S2"): 0.0011, ("DPH", "under 1M", "S2"): 0.0023, ("DPH", "under 1M", "S1"): -0.0007}
    for (m, g, r), v in rep.items():
        check("T7", f"{m} {g} {r} - draw mean vs reference constant (Table 6)", round(df[(df.base == m) & (df.group == g) & (df.rule == r)].difference_ndcg10.iloc[0], 4), v, 1e-9)
    return df


# --------------------------------------------------------------------------------------------------------------- T8
def t8():
    import lib_direct as RD
    r = pd.read_csv(f"{SL}/recheck/risk_by_cell.csv")
    r = r[r.sidecar == "S2"].copy()
    lev = {}
    for g, corp in (("under_1M", RD.SMALL), ("over_1M", RD.BIG)):
        b = pd.concat([pd.read_parquet(f"{SL}/ours_tuned/grid_{ds}.parquet", columns=["dataset", "qid", "model", "arm", "ndcg10"],
                                       filters=[("arm", "==", "base")]) for ds in corp])
        b = b.drop_duplicates(["dataset", "qid", "model"])
        for model in ("bm25", "dph"):
            lev[(g, model)] = macro_level(b[b.model == model])
    df = pd.DataFrame({"base": r.model.str.upper(), "group": r.group.map(GROUP_NAME), "K": r.K, "w": r.w,
                       "crve_default": (r.K == 5) & np.isclose(r.w, 0.2), "collections": r.corpora,
                       "base_ndcg10_macro": [lev[(g, m)] for g, m in zip(r.group, r.model)], "macro_gain": r.macro_gain,
                       "collections_below_-0.005": r["corpora_below_-0.005"], "queries": r.queries})
    df["macro_gain_rel_pct"] = [rel(a, b) for a, b in zip(df.macro_gain, df.base_ndcg10_macro)]
    df["notes"] = "S2 sidecar; capped query sets of ours_tuned/ (at most 500 queries per collection); point estimates"
    df = df.sort_values(["base", "group", "K", "w"], ascending=[True, False, True, True]).reset_index(drop=True)
    write("T8", df, delta_cols=["macro_gain"], level_cols=["base_ndcg10_macro"],
          pct_cols=["macro_gain_rel_pct"], int_cols=["K", "collections", "collections_below_-0.005", "queries"],
          fixed_cols={"w": 2})
    # reference constants: the values printed in Table A4 (rounded to 4 decimals)
    for (m, g, w, v) in (("BM25", "under 1M", 0.2, 0.0091), ("BM25", "under 1M", 0.1, 0.0069), ("DPH", "over 1M", 0.2, -0.0001), ("BM25", "over 1M", 0.2, 0.0044)):
        x = df[(df.base == m) & (df.group == g) & (df.K == 5) & np.isclose(df.w, w)].macro_gain.iloc[0]
        check("T8", f"{m} {g} K=5 w={w} vs reference constant (Table A4)", round(x, 4), v, 1e-9)
    return df


# ------------------------------------------------------------------------------------------- pool size and weighting
def pool_and_weighting():
    """Section 5.3, offline settings: the pool-size line (computed from results/pool_reduced/) and the per-term weighting line
    (from results/recheck/full_queries/verdict.json)."""
    P = f"{SL}/pool_reduced"
    rv = json.load(open(f"{P}/rep_verdict.json"))
    rx = json.load(open(f"{P}/rep_extra.json"))
    c1 = rv["condition_1_macro_90pct"]
    kept = c1["a030_gain"] / c1["full_gain"]  # ratio of the replicate-mean gains over BM25 (five builds each)
    check("pool", "kept share = ratio of the stored replicate-mean gains", kept, c1["kept"], 1e-12)
    lo, hi = rx["kept_share_bootstrap"]["ci95"]
    per = {c: v["kept"] for c, v in rv["condition_2_per_corpus_half"]["corpora"].items()}
    low = sorted((c for c, k in per.items() if k < 0.5), key=lambda c: per[c])
    pc = lambda x: f"{100 * x:.0f}%"
    names = ", ".join(NAME[c] for c in low[:-1]) + f" and {NAME[low[-1]]}"
    v = json.load(open(f"{SL}/recheck/full_queries/verdict.json"))["secondaries"]
    s1, s2 = v["S1"], v["S2"]
    f = lambda x: f"{x:+.4f}".replace("-", "−")
    lines = [
        "# Pool size and per-term weighting (Section 5.3)",
        "",
        "Written by `paper/s52_53.py`.",
        "",
        f"**Pool size.** A pool of 30% of the eligible terms, rebuilt in full (five builds, against five builds of the full pool), "
        f"keeps {pc(kept)} of the gain in nDCG@10 over BM25 (ratio of the mean gains of the five builds; query-bootstrap interval "
        f"{pc(lo)} to {pc(hi)}); {names} keep only {pc(per[low[0]])} to {pc(per[low[-1]])} of their own gains. Measured before "
        "the evaluation corrections (ArguAna included), with randomly drawn chunks (seed 42), the two-channel Gate 1, K = 5 and "
        "w = 0.20. "
        "Sources: `results/pool_reduced/rep_verdict.json`, `rep_extra.json`.",
        "",
        f"**Per-term weighting.** Weighting each term by the certainty of its chunk sample, against the flat weight w = 0.20 "
        f"(randomly drawn chunks, the 30,541 queries of Section 5.3): under 1M {f(s1['macro'])}, Holm-adjusted "
        f"{lvl(s1['level'])} interval [{f(s1['interval'][0])}, {f(s1['interval'][1])}]; over 1M {f(s2['macro'])}, "
        f"{lvl(s2['level'])} interval [{f(s2['interval'][0])}, {f(s2['interval'][1])}]. "
        "Source: `results/recheck/full_queries/verdict.json`.",
        "",
    ]
    # reference constants: the values printed in Section 5.3
    check("pool", "kept share (about 85%)", round(100 * kept), 85, 0)
    check("pool", "interval of the kept share (66% to 103%)", f"{round(100 * lo)}-{round(100 * hi)}", "66-103")
    check("pool", "collections keeping less than half (FiQA, Sustainable Living, ArguAna: 36% to 49%)",
          f"{';'.join(low)} {round(100 * per[low[0]])}-{round(100 * per[low[-1]])}", "fiqa;bright_sustainable_living;arguana 36-49")
    check("weighting", "under 1M certainty - flat (+0.0005)", round(s1["macro"], 4), 0.0005, 1e-9)
    check("weighting", "under 1M interval (Holm 97.5%: -0.0031, +0.0047)",
          f"{lvl(s1['level'])} {s1['interval'][0]:+.4f} {s1['interval'][1]:+.4f}", "97.5% -0.0031 +0.0047")
    check("weighting", "over 1M certainty - flat (-0.0008)", round(s2["macro"], 4), -0.0008, 1e-9)
    check("weighting", "over 1M interval (95%: -0.0021, +0.0004)",
          f"{lvl(s2['level'])} {s2['interval'][0]:+.4f} {s2['interval'][1]:+.4f}", "95% -0.0021 +0.0004")
    with open(os.path.join(OUT, "pool_and_weighting.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("pool_and_weighting.md written", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t5()
    t5b()
    t6()
    f2()
    f3()
    t7()
    t8()
    pool_and_weighting()
    c = pd.DataFrame(CHECKS)
    c.to_csv(os.path.join(OUT, "checks_s52_53.csv"), index=False)
    print(f"checks: {int(c.ok.sum())} of {len(c)} ok", flush=True)
    if (~c.ok).any():
        print(c[~c.ok].to_string(index=False), flush=True)
        raise SystemExit(f"{int((~c.ok).sum())} cross-check(s) failed")


if __name__ == "__main__":
    main()
