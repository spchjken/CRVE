#!/usr/bin/env python3
"""Table 1 of the paper (collections): documents, evaluated queries and S2 sidecar coverage.

  python scripts/table1.py [--out results/runs/table1.csv] [--sidecar-root data/cache/sidecars/S2] [--raw]

Documents: results/collection_stats.csv (num.Documents of each Terrier index). Queries: results/queries/<c>.txt (the
evaluated queries; the same sets as the base arm of results/feature_sidecar/outcomes_<c>.parquet, which is checked).
Coverage: share of documents with at least one stored chunk in the S2 sidecar, from
results/recheck/coverage_documents_with_chunk.csv; with --sidecar-root it is recounted from the sidecars' term references
(distinct document ids). With --raw, the judged queries are counted from data/raw and the queries left out of each
collection are checked against the table's notes (50 per collection marked 1, 450 in all; 544 for the collections
marked 3). The computed table is compared with the numbers printed in the paper; any difference stops with exit status 1.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd

NAMES = {"scifact": ("SciFact", "BEIR"), "nfcorpus": ("NFCorpus", "BEIR"), "trec_covid": ("TREC-COVID", "BEIR"), "fiqa": ("FiQA", "BEIR"),
         "quora": ("Quora", "BEIR"), "arguana": ("ArguAna", "BEIR"), "scidocs": ("SCIDOCS", "BEIR"), "webis_touche2020": ("Touché-2020", "BEIR"),
         "bright_aops": ("AoPS", "BRIGHT"), "bright_economics": ("Economics", "BRIGHT"), "bright_leetcode": ("LeetCode", "BRIGHT"),
         "bright_stackoverflow": ("Stack Overflow", "BRIGHT"), "bright_sustainable_living": ("Sustainable Living", "BRIGHT"),
         "bright_biology": ("Biology", "BRIGHT"), "bright_earth_science": ("Earth Science", "BRIGHT"), "bright_psychology": ("Psychology", "BRIGHT"),
         "bright_robotics": ("Robotics", "BRIGHT"), "bright_pony": ("Pony", "BRIGHT"), "bright_theoremqa_questions": ("TheoremQA-Questions", "BRIGHT"),
         "bright_theoremqa_theorems": ("TheoremQA-Theorems", "BRIGHT"), "nq": ("NQ", "BEIR"), "dbpedia_entity": ("DBPedia-Entity", "BEIR"),
         "hotpotqa": ("HotpotQA", "BEIR"), "fever": ("FEVER", "BEIR"), "climate_fever": ("Climate-FEVER", "BEIR")}
# as printed in Table 1 (order of the table): documents, queries, note on the queries, coverage in percent
PAPER = [("scifact", 5183, 250, 1, 100), ("nfcorpus", 3633, 273, 1, 100), ("trec_covid", 171331, 50, 2, 49), ("fiqa", 57600, 598, 1, 84),
         ("quora", 522931, 9950, 1, 33), ("arguana", 8674, 1310, 3, 100), ("scidocs", 25657, 902, 3, 97), ("webis_touche2020", 382545, 49, 2, 26),
         ("bright_aops", 188002, 61, 1, 50), ("bright_economics", 50220, 53, 1, 64), ("bright_leetcode", 413932, 92, 1, 30),
         ("bright_stackoverflow", 107081, 67, 1, 43), ("bright_sustainable_living", 60792, 58, 1, 64), ("bright_biology", 57359, 53, 3, 79),
         ("bright_earth_science", 121249, 66, 3, 61), ("bright_psychology", 52835, 51, 3, 67), ("bright_robotics", 61961, 51, 3, 52),
         ("bright_pony", 7894, 62, 3, 73), ("bright_theoremqa_questions", 188002, 144, 3, 50), ("bright_theoremqa_theorems", 23839, 26, 3, 77),
         ("nq", 2681468, 3452, None, 6), ("dbpedia_entity", 4635922, 400, None, 3), ("hotpotqa", 5233329, 7405, None, 3),
         ("fever", 5416568, 6666, None, 3), ("climate_fever", 5416593, 1535, None, 3)]
TOTALS = {"all": 33624, "under_1M": 14166, "over_1M": 19458, "left_out": 994, "left_out_note1": 450, "left_out_note3": 544}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--sidecar-root", default=None)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    from crve.pipeline import common as C
    stats = pd.read_csv("results/collection_stats.csv").set_index("collection")
    cov = pd.read_csv("results/recheck/coverage_documents_with_chunk.csv").set_index("dataset")
    rows, problems = [], []
    for ds, docs_p, q_p, note, cov_p in PAPER:
        q = C.queries(ds)
        o = pd.read_parquet(f"results/feature_sidecar/outcomes_{ds}.parquet", columns=["qid", "model", "arm"])
        if set(q) != set(o[(o.arm == "base") & (o.model == "bm25")].qid):
            problems.append(f"{ds}: results/queries differs from the stored outcomes")
        share = float(cov.loc[ds, "S2_share"])
        if a.sidecar_root:
            import pyarrow.parquet as pq
            tr = pq.read_table(os.path.join(a.sidecar_root, ds, "term_references.parquet"), columns=["doc_ids"]).column("doc_ids").to_pylist()
            n_cov = len({str(x) for ids in tr for x in ids})
            if n_cov != int(cov.loc[ds, "S2_documents_with_chunk"]):
                problems.append(f"{ds}: {n_cov} documents with a stored chunk, stored count {int(cov.loc[ds, 'S2_documents_with_chunk'])}")
            share = n_cov / int(stats.loc[ds, "num_documents"])
        r = {"collection": NAMES[ds][0], "benchmark": NAMES[ds][1], "group": "over_1M" if ds in C.OVER_1M else "under_1M",
             "documents": int(stats.loc[ds, "num_documents"]), "queries": len(q), "note": note, "coverage_percent": round(100 * share)}
        if a.raw:
            from evaluation.benchmark_loader import BenchmarkLoader
            qs, _ = BenchmarkLoader.load_queries(ds)
            r["judged"] = len(qs)
            r["left_out"] = len(qs) - len(q)
            if not set(q) <= {str(x["query_id"]) for x in qs}:
                problems.append(f"{ds}: evaluated queries that are not judged")
        rows.append(r)
        for k, v in (("documents", docs_p), ("queries", q_p), ("coverage_percent", cov_p)):
            if r[k] != v:
                problems.append(f"{ds}: {k} {r[k]} vs {v} in the paper")
    t = pd.DataFrame(rows)
    t["note"] = t["note"].astype("Int64")
    tot = {"all": int(t.queries.sum()), "under_1M": int(t[t.group == "under_1M"].queries.sum()), "over_1M": int(t[t.group == "over_1M"].queries.sum())}
    if a.raw:
        tot.update({"left_out": int(t.left_out.sum()), "left_out_note1": int(t[t.note == 1].left_out.sum()),
                    "left_out_note3": int(t[t.note == 3].left_out.sum())})
        if (t[t.note == 1].left_out != 50).any() or (t[t.note == 2].left_out != 0).any() or (t[t.group == "over_1M"].left_out != 0).any():
            problems.append("left-out queries do not follow the notes of Table 1")
    for k, v in tot.items():
        if v != TOTALS[k]:
            problems.append(f"total {k} {v} vs {TOTALS[k]}")
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    print(tot)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out, index=False)
    if problems:
        raise SystemExit("Table 1 differs:\n" + "\n".join(problems))
    print("Table 1: all numbers reproduced")


if __name__ == "__main__":
    main()
