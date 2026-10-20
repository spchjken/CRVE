#!/usr/bin/env python3
"""Recompute the worked example of Figure 1: FiQA query q_beir_fiqa_2713 ("Physical Checks - Mailing") through CRVE.

  python scripts/figure1_example.py [--out results/runs/figure1_fiqa_2713.csv]

Needs the fiqa Terrier index, the stored Gate 1 assets (data/cache/gate1/fiqa_*) and the stored S2 sidecar
(data/cache/sidecars/S2/fiqa). Prints the Gate 1 top 10 with both channel ranks and RRF scores, the Gate 2 cosine of every
candidate (the maximum over its stored chunks) with its closest stored chunk, the Gate 2 top 5, and the BM25 nDCG@10 of the
base and expanded query (live retrieval). The numbers printed in the figure are checked; any difference stops with exit
status 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd

# printed in Figure 1
FIG_GATE1_TOP6 = ["mailbox", "victoria", "capitalon", "nevada", "verif", "postag"]
FIG_GATE1_RANKS = {"electron": 24, "bearer": 70}
FIG_GATE2_TOP5 = [("mailbox", 0.795), ("postag", 0.770), ("verif", 0.763), ("electron", 0.759), ("bearer", 0.755)]
FIG_DROPPED = [("victoria", 0.641), ("capitalon", 0.633), ("nevada", 0.627)]
FIG_SIXTH = 0.752
FIG_NDCG = (0.193, 0.307)
FIG_CHUNKS = {"mailbox": "not writing a check and sticking a stamp and then putting it in a mailbox",
              "capitalon": "Both Scottrade and ING Direct (CapitalOne) have physical branches"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="fiqa")
    ap.add_argument("--qid", default="q_beir_fiqa_2713")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from crve.pipeline import common as C
    from crve.pipeline import gate1 as G
    from crve.pipeline import gate2 as G2
    from crve.pipeline import retrieval as R
    from crve.selection.gate2_sidecar import ContextSidecar
    from crve.scoring.context_evidence import encode_queries
    from crve.scoring.ranking import ranked_list
    from evaluation.benchmark_loader import BenchmarkLoader
    C.pin_terrier()
    ds, qid = a.corpus, a.qid
    qtext = BenchmarkLoader.load_query_texts_only(ds)[qid]

    # Gate 1: the two channels (pool terms, query terms removed), RRF k = 60, top 200
    ch = G.channel_lists(ds, [qid], G.profiles_dir(ds), 42)[qid]
    cands = ch["rrf"]
    if cands != G.candidates(ds, [qid], G.profiles_dir(ds), 42)[qid]:
        raise SystemExit("channel_lists and candidates disagree")
    rank_p = {t: i + 1 for i, (t, _) in enumerate(ch["PPMISidecar"])}
    rank_s = {t: i + 1 for i, (t, _) in enumerate(ch["SparseLexicalContextProfiles"])}

    # Gate 2: max cosine over each candidate's stored S2 chunks
    side = C.sidecar_dir(ds)
    sc = ContextSidecar(side, verify_checksums=True)
    texts = pd.read_parquet(os.path.join(side, "chunks.parquet"), columns=["text"]).text.tolist()
    qv = encode_queries({(ds, qid): qtext}, device="cpu", seed=42)[(ds, qid)].astype(np.float32)
    rows = []
    for r, t in enumerate(cands, 1):
        m = sc.get_term_metadata(t)
        row = {"term": t, "gate1_rank": r, "ppmi_rank": rank_p.get(t), "profile_rank": rank_s.get(t),
               "rrf_score": sum(1.0 / (G.RRF_K + x) for x in (rank_p.get(t), rank_s.get(t)) if x is not None)}
        if m and m["status"] == "VERIFIED" and m["sample_n"] > 0:
            ci = [int(i) for i in m["chunk_indices"]]
            cos = np.clip(np.asarray(sc.embeddings[ci], dtype=np.float32) @ qv, -1.0, 1.0)
            j = int(np.argmax(cos))
            row.update({"gate2_cosine": float(cos[j]), "best_chunk_position": j + 1, "stored_chunks": len(ci),
                        "chunks_with_term": int(m["population_N"]), "best_chunk_doc": str(m["doc_ids"][j]), "best_chunk": texts[ci[j]]})
        rows.append(row)
    df = pd.DataFrame(rows)
    scaled = np.clip((df.gate2_cosine.to_numpy(float) + 1.0) / 2.0, 0.0, 1.0)  # the score Gate 2 sorts by
    order = ranked_list(df.term.tolist(), scaled, df.gate1_rank.to_numpy(float), top=len(df))
    df["gate2_rank"] = df.term.map({t: i + 1 for i, t in enumerate(order)})
    top5 = G2.top5(ds, {qid: cands}, side)[qid]
    if top5 != order[:5]:
        raise SystemExit(f"Gate 2 order differs from top5(): {order[:5]} vs {top5}")

    # retrieval: base query and the expanded query, BM25 and DPH with termpipelines=NoOp
    R.patch_path_f()
    sets = pd.DataFrame([{"dataset": ds, "qid": qid, "arm": "S2", "terms": list(top5)}])
    out = R.live_outcomes(ds, sets)
    nd = out.set_index(["model", "arm"]).ndcg10

    pd.set_option("display.width", 200)
    print(f"query {qid}: {qtext!r}")
    print("Gate 1 top 10 (+ the Gate 2 picks below):")
    show = df[(df.gate1_rank <= 10) | df.term.isin(top5)].sort_values("gate1_rank")
    print(show[["gate1_rank", "term", "ppmi_rank", "profile_rank", "rrf_score", "gate2_rank", "gate2_cosine"]].round(5).to_string(index=False))
    print("Gate 2 top 7:")
    g2 = df.sort_values("gate2_rank").head(7)
    print(g2[["gate2_rank", "term", "gate1_rank", "gate2_cosine", "best_chunk_position", "stored_chunks", "chunks_with_term", "best_chunk_doc"]].round(4).to_string(index=False))
    for t in ["mailbox", "capitalon"]:
        print(f"closest stored chunk of {t}: {df.set_index('term').loc[t, 'best_chunk']!r}")
    print(f"BM25 nDCG@10: {nd[('bm25', 'base')]:.4f} -> {nd[('bm25', 'S2')]:.4f}; DPH: {nd[('dph', 'base')]:.4f} -> {nd[('dph', 'S2')]:.4f}")

    # the figure's numbers
    by = df.set_index("term")
    checks = {
        "gate1_top6": (cands[:6], FIG_GATE1_TOP6),
        "gate1_ranks": ({t: int(by.loc[t, "gate1_rank"]) for t in FIG_GATE1_RANKS}, FIG_GATE1_RANKS),
        "gate2_top5": ([(t, round(float(by.loc[t, "gate2_cosine"]), 3)) for t in top5], FIG_GATE2_TOP5),
        "dropped": ([(t, round(float(by.loc[t, "gate2_cosine"]), 3)) for t, _ in FIG_DROPPED], FIG_DROPPED),
        "sixth": (round(float(g2.iloc[5].gate2_cosine), 3), FIG_SIXTH),
        "ndcg10_bm25": ((round(float(nd[("bm25", "base")]), 3), round(float(nd[("bm25", "S2")]), 3)), FIG_NDCG),
        "chunks": ({t: s in by.loc[t, "best_chunk"] for t, s in FIG_CHUNKS.items()}, {t: True for t in FIG_CHUNKS}),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    print(json.dumps({k: {"computed": v[0], "figure": v[1]} for k, v in checks.items()}, default=str))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.sort_values("gate1_rank").to_csv(a.out, index=False)
    if bad:
        raise SystemExit(f"differs from Figure 1: {sorted(bad)}")
    print("Figure 1: all numbers reproduced")


if __name__ == "__main__":
    main()
