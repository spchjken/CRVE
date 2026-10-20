#!/usr/bin/env python3
"""
Figure F1 split by corpus size (the paper's Figures 2 and 3). Same data and style as F1 (s51.py); read-only, no retrieval.

  python paper/f1_split.py   (from the repository root, after s51.py; or run everything with python paper/make_all.py)

Reads paper/output/F1_data.csv (written by s51.py; not recomputed) and writes
  F1_under1M.pdf (20 collections under 1M documents, 12 cm wide) and F1_over1M.pdf (5 collections over 1M, 6 cm wide; both placed inside a text column, so rendered natively at that size),
  paper/output/F1_under1M_data.csv and F1_over1M_data.csv (+ _rounded).
Collections are ordered by the number of documents, ascending (ties: collection name), NOT by gain.
Document counts and size columns come from the frozen file results/collection_stats.csv: num_documents is num.Documents of each
collection's Terrier index (data.properties), cross-checked here against the counts printed in Table 1 of the paper.
Size columns (not used for the order): raw_corpus_bytes (BEIR: size of the collection's corpus.jsonl; BRIGHT: UTF-8 bytes of the
'content' column of the collection's documents parquet file, i.e. text only, no ids or keys) and terrier_index_bytes (size on disk
of the Terrier index directory).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import (MARKERS, NAME, OUT, OVER, RES, UNDER, WIDTH_12_CM, WIDTH_6_CM, new_fig, save_fig, setup_matplotlib, write)

FLOOR = 0.005
COLOR = {"bm25": "#005AA0", "dph": "#8a8a8a"}  # accent blue and mid grey; marker shape (filled circle / open square) is the secondary encoding
BAND = "#edf3f9"
ERR_ALPHA = 0.7
STATS = os.path.join(RES, "collection_stats.csv")  # frozen: document counts and sizes per collection
F1_COLS = ["queries", "base_ndcg10", "crve_s2_ndcg10", "gain", "ci_low", "ci_high", "relative_gain_pct"]

# Reference constants: the document counts printed in Table 1 of the paper, by display name.
TABLE1_DOCUMENTS = {"SciFact": 5183, "NFCorpus": 3633, "TREC-COVID": 171331, "FiQA": 57600, "Quora": 522931, "ArguAna": 8674,
                    "SCIDOCS": 25657, "Touché-2020": 382545, "AoPS": 188002, "Economics": 50220, "LeetCode": 413932,
                    "Stack Overflow": 107081, "Sustainable Living": 60792, "Biology": 57359, "Earth Science": 121249,
                    "Psychology": 52835, "Robotics": 61961, "Pony": 7894, "TheoremQA-Questions": 188002, "TheoremQA-Theorems": 23839,
                    "NQ": 2681468, "DBPedia-Entity": 4635922, "HotpotQA": 5233329, "FEVER": 5416568, "Climate-FEVER": 5416593}


def count_label(n):
    """5183 -> 5.2k, 57359 -> 57k, 171331 -> 171k, 2681468 -> 2.7M."""
    if n < 1000:
        return str(n)
    if n < 100_000:
        v = n / 1000
        return f"{v:.1f}k" if v < 10 else f"{v:.0f}k"
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1e6:.1f}M"


def sizes():
    st = pd.read_csv(STATS).set_index("collection")
    rows = []
    for ds in UNDER + OVER:
        n = int(st.loc[ds, "num_documents"])
        if TABLE1_DOCUMENTS.get(NAME[ds]) != n:
            raise SystemExit(f"document count mismatch for {ds}: collection_stats.csv {n}, Table 1 {TABLE1_DOCUMENTS.get(NAME[ds])}")
        rows.append({"dataset": ds, "documents": n, "raw_corpus_bytes": int(st.loc[ds, "raw_corpus_bytes"]),
                     "terrier_index_bytes": int(st.loc[ds, "terrier_index_bytes"])})
    print(f"document counts: collection_stats.csv equals Table 1 for all {len(rows)} collections", flush=True)
    return pd.DataFrame(rows)


def build(f1, sz, group_list, group):
    d = f1[f1.dataset.isin(group_list)].merge(sz, on="dataset")
    order = d[d.base == "BM25"].sort_values(["documents", "collection"]).dataset.tolist()
    rank = {ds: i + 1 for i, ds in enumerate(order)}
    d["order"] = d.dataset.map(rank).astype(int)
    d["tick_label"] = [f"{NAME[ds]} ({count_label(n)})" for ds, n in zip(d.dataset, d.documents)]
    d = d.sort_values(["order", "base"]).reset_index(drop=True)
    cols = ["order", "dataset", "collection", "group", "base", "documents", "tick_label", "raw_corpus_bytes", "terrier_index_bytes"] + F1_COLS
    return d[cols], order


def verify(f1, written_id, d):
    """The written CSV must reproduce F1_data.csv exactly (unrounded)."""
    w = pd.read_csv(f"{OUT}/{written_id}.csv", float_precision="round_trip")
    m = w.merge(f1, on=["dataset", "base"], suffixes=("", "_f1"), how="left", indicator=True)
    if len(m) != len(d) or (m["_merge"] != "both").any():
        raise SystemExit(f"{written_id}: rows not found in F1_data.csv")
    for c in F1_COLS:
        a, b = m[c].to_numpy(dtype=float), m[c + "_f1"].to_numpy(dtype=float)
        if not np.array_equal(a, b, equal_nan=True):
            raise SystemExit(f"{written_id}: column {c} differs from F1_data.csv")
    print(f"{written_id}: F1 values equal F1_data.csv exactly ({len(m)} rows)", flush=True)


# plot-only short names for the tick labels at the small native size (the CSV tick_label column keeps the full names)
SHORT = {"TheoremQA-Theorems": "TheoremQA-Thm.", "TheoremQA-Questions": "TheoremQA-Q.", "Sustainable Living": "Sust. Living"}
FS_TICK, FS_AXIS, FS_LEGEND = 6.5, 7.5, 7.0


def short_tick(label):
    for k, v in SHORT.items():
        label = label.replace(k, v)
    return label


def figure(d, order, fig_id, width_cm, height_cm, ylim=None, legend_ncol=3, rot=55, legend_above=True, tick_fs=FS_TICK, xpad=0.7, pair_off=0.17, two_line_ylabel=False):
    plt = setup_matplotlib()
    fig, ax = new_fig(width_cm, height_cm)
    n = len(order)
    xpos = {ds: i for i, ds in enumerate(order)}
    ax.axhspan(-FLOOR, FLOOR, color=BAND, zorder=0, lw=0, label="within ±0.005")
    ax.axhline(0, color="#4d4d4d", lw=0.6, zorder=1)
    off = {"bm25": -pair_off, "dph": pair_off}
    for model, lab in (("bm25", "over BM25"), ("dph", "over DPH")):
        s = d[d.base == model.upper()].set_index("dataset").loc[order]
        x = np.array([xpos[k] for k in order]) + off[model]
        col = COLOR[model]
        eb = ax.errorbar(x, s.gain, yerr=[s.gain - s.ci_low, s.ci_high - s.gain], fmt=MARKERS[model], color=col,
                         mfc=col if model == "bm25" else "white", mec=col, ecolor=col,
                         elinewidth=0.5, capsize=1.2, capthick=0.5, ms=3.5, lw=0, zorder=3, label=f"CRVE {lab}")
        for art in list(eb[1]) + list(eb[2]):  # caps and bars only; the markers stay opaque
            art.set_alpha(ERR_ALPHA)
    ax.set_xlim(-xpad, n - 1 + xpad)
    if ylim:
        ax.set_ylim(*ylim)
    lab = d[d.base == "BM25"].set_index("dataset").loc[order].tick_label
    ax.set_xticks(range(n))
    ax.set_xticklabels([short_tick(t) for t in lab], rotation=rot, ha="right", rotation_mode="anchor", fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Gain in nDCG@10 over the base" if not two_line_ylabel else "Gain in nDCG@10\nover the base", fontsize=FS_AXIS)
    if legend_above:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=legend_ncol, columnspacing=1.0, handletextpad=0.4, fontsize=FS_LEGEND)
    else:  # row(s) above the axes, space reserved by the constrained layout
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", frameon=False, ncol=legend_ncol, columnspacing=0.8, handletextpad=0.3, handlelength=1.2, labelspacing=0.3, fontsize=FS_LEGEND)
    save_fig(fig, fig_id)
    plt.close(fig)


def main():
    f1 = pd.read_csv(f"{OUT}/F1_data.csv", float_precision="round_trip")
    sz = sizes()
    print(sz.assign(raw_MB=lambda x: x.raw_corpus_bytes / 1e6, index_MB=lambda x: x.terrier_index_bytes / 1e6)
          .drop(columns=["raw_corpus_bytes", "terrier_index_bytes"]).round(1).to_string(index=False), flush=True)
    specs = [("under_1M", UNDER, "F1_under1M", WIDTH_12_CM, 6.5, 3, 50, None, True), ("over_1M", OVER, "F1_over1M", WIDTH_6_CM, 5.5, 2, 55, None, False)]
    for group, lst, fig_id, w, h, ncol, rot, ylim, above in specs:
        d, order = build(f1, sz, lst, group)
        write(f"{fig_id}_data", d, delta_cols=["gain", "ci_low", "ci_high"], level_cols=["base_ndcg10", "crve_s2_ndcg10"],
              pct_cols=["relative_gain_pct"], int_cols=["order", "documents", "raw_corpus_bytes", "terrier_index_bytes", "queries"])
        verify(f1, f"{fig_id}_data", d)
        figure(d, order, fig_id, w, h, legend_ncol=ncol, rot=rot, ylim=ylim, legend_above=above, **({} if above else dict(tick_fs=FS_TICK, xpad=0.45, pair_off=0.2, two_line_ylabel=True)))
        print(f"{fig_id} order: " + ", ".join(f"{NAME[k]} ({count_label(int(sz.set_index('dataset').documents[k]))})" for k in order), flush=True)


if __name__ == "__main__":
    main()
