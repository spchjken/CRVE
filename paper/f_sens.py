#!/usr/bin/env python3
"""
F_sens: one full-width sensitivity figure for Section 5.3 that combines the data of F2 (Gate 1 depth L), F3 (chunk grid) and T8 (K x w).
Reads only the stored data files that s52_53.py wrote (F2_data.csv, F3_data.csv, T8.csv); nothing is recomputed. The F2, F3 and T8 outputs
are left as they are.

  python paper/f_sens.py     (from the repository root, after s52_53.py; or run everything with python paper/make_all.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, WIDTH_FULL_CM, save_fig, setup_matplotlib, write  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HEIGHT_CM = 12.0
MEAN = "mean of S1 and S2 (per query)"


def fail(msg):
    raise SystemExit(f"f_sens: {msg}")


def load():
    f2 = pd.read_csv(os.path.join(OUT, "F2_data.csv"))
    f3 = pd.read_csv(os.path.join(OUT, "F3_data.csv"))
    t8 = pd.read_csv(os.path.join(OUT, "T8.csv"))
    if "collections_below_-0.005" not in t8:
        fail("collections_below_-0.005 column missing")
    t8.rename(columns={"collections_below_-0.005": "below"}, inplace=True)
    return f2, f3, t8


def build_data(f2, f3, t8):
    """One long table with every value drawn, copied from F2_data, F3_data and T8 without change."""
    rows = []
    a = f2[(f2.group == "under 1M") & f2.variant.isin(["S2", MEAN])]
    for r in a.itertuples(index=False):
        rows.append({"panel": "a", "source": "F2_data", "group": r.group, "base": r.base, "series": "S2 alone" if r.variant == "S2" else r.variant,
                     "L": r.L, "gain_ndcg10": r.gain_ndcg10, "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "gain_rel_pct": r.gain_rel_pct,
                     "notes": r.notes})
    b = f3[f3.K == 5]
    for r in b.itertuples(index=False):
        conf = r.sidecar.startswith("CRVE setting")
        rows.append({"panel": "b", "source": "F3_data", "group": "under 1M (18 collections)", "base": "BM25",
                     "series": "CRVE setting (star)" if conf else f"chunk length {int(r.chunk_length_tokens)} tokens", "K": r.K,
                     "chunk_length_tokens": r.chunk_length_tokens, "chunks_per_term": r.chunks_per_term, "gain_ndcg10": r.gain_ndcg10_over_bm25,
                     "gain_rel_pct": r.gain_rel_pct, "notes": r.notes})
    for r in t8.itertuples(index=False):
        rows.append({"panel": "c", "source": "T8", "group": r.group, "base": r.base, "series": f"K = {int(r.K)}", "K": r.K, "w": r.w,
                     "crve_default": bool(r.crve_default), "gain_ndcg10": r.macro_gain, "gain_rel_pct": r.macro_gain_rel_pct,
                     "collections_below_-0.005": r.below, "collections": r.collections, "notes": r.notes})
    cols = ["panel", "source", "group", "base", "series", "L", "K", "chunk_length_tokens", "chunks_per_term", "w", "crve_default", "gain_ndcg10",
            "ci95_low", "ci95_high", "gain_rel_pct", "collections_below_-0.005", "collections", "notes"]
    df = pd.DataFrame(rows).reindex(columns=cols)
    df["crve_default"] = df.crve_default.astype("boolean")
    return df


def checks(f2, f3, t8, df):
    """The values drawn must equal the sources (and reference constants read off Figure 4)."""
    want = {("BM25", "S2 alone", 200): 0.0090, ("BM25", "S2 alone", 500): 0.0115, ("BM25", MEAN, 200): 0.0093, ("BM25", MEAN, 500): 0.0098,
            ("DPH", "S2 alone", 200): 0.0112, ("DPH", "S2 alone", 500): 0.0124, ("DPH", MEAN, 200): 0.0097, ("DPH", MEAN, 500): 0.0102}
    a = df[df.panel == "a"]
    for (base, series, L), v in want.items():
        x = a[(a.base == base) & (a.series == series) & (a.L == L)].gain_ndcg10
        if len(x) != 1 or round(float(x.iloc[0]), 4) != v:
            fail(f"(a) {base} {series} L={L}: {x.tolist()} != {v}")
    star = df[(df.panel == "b") & (df.series == "CRVE setting (star)")]
    if len(star) != 1 or round(float(star.gain_ndcg10.iloc[0]), 4) != 0.0071:
        fail("(b) CRVE setting is not +0.0071")
    c = df[(df.panel == "c") & df.crve_default.fillna(False).astype(bool)]
    if len(c) != 4 or not (np.isclose(c.K, 5).all() and np.isclose(c.w, 0.2).all()):
        fail("(c) default cells are not the four K = 5, w = 0.20 cells")
    if (len(a), len(df[df.panel == "b"]), len(df[df.panel == "c"])) != (28, 31, 48):
        fail("unexpected row counts")
    print("checks: ok", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------- drawing
# Colours: accent blue #005AA0 / mid grey #8a8a8a for the two categories; one blue ramp (light to dark) for the ordered series; black highlights.
# Marker shapes, open/filled markers and line styles stay as before so the figure still reads in greyscale.
A_STYLE = {"S2 alone": dict(color="#005AA0", marker="o", linestyle="-", markerfacecolor="#005AA0", zorder=3, lw=1.1),
           MEAN: dict(color="#8a8a8a", marker="s", linestyle="--", markerfacecolor="white", zorder=2, lw=0.9)}
ERR_ALPHA = 0.7
HIGHLIGHT = "#000000"
B_STYLE = {32: ("#86b0da", "^", ":"), 64: ("#5a90c6", "s", "--"), 100: ("#2f73b1", "o", "-"), 128: ("#005AA0", "D", "-."),
           256: ("#003a6b", "v", (0, (5, 1.5, 1, 1.5, 1, 1.5)))}
C_STYLE = {1: ("#86b0da", "v", ":"), 3: ("#4f88c2", "s", "--"), 5: ("#005AA0", "o", "-"), 10: ("#003a6b", "D", "-.")}
C_DODGE = {1: -0.039, 3: -0.013, 5: 0.013, 10: 0.039}


def panel_a(sf, df):
    axes = sf.subplots(1, 2, sharey=True)
    a = df[df.panel == "a"]
    for ax, base in zip(axes, ("BM25", "DPH")):
        for series, st in A_STYLE.items():
            p = a[(a.base == base) & (a.series == series)].sort_values("L")
            off = 1.035 if series == MEAN else 1 / 1.035
            eb = ax.errorbar(p.L * off, p.gain_ndcg10, yerr=[p.gain_ndcg10 - p.ci95_low, p.ci95_high - p.gain_ndcg10], color=st["color"],
                        marker=st["marker"], linestyle=st["linestyle"], markerfacecolor=st["markerfacecolor"], lw=st["lw"], elinewidth=0.45,
                        capsize=1.5, capthick=0.45, ecolor=st["color"], zorder=st["zorder"],
                        label="S2 alone" if series == "S2 alone" else "Mean of S1 and S2")
            for art in list(eb[1]) + list(eb[2]):  # caps and bars only; the markers stay opaque
                art.set_alpha(ERR_ALPHA)
        ax.axhline(0, color="#4d4d4d", linewidth=0.6)
        ax.axvline(200, color="#7a7a7a", linewidth=0.7, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100, 200, 500])
        ax.set_xticklabels(["10", "25", "50", "100", "200", "500"])
        ax.minorticks_off()
        ax.set_xlabel("Gate 1 depth L")
        ax.text(0.03, 0.97, f"Over {base}", transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    axes[0].set_ylabel("Gain in nDCG@10")
    axes[0].set_ylim(-0.0025, 0.0175)
    h, l = axes[0].get_legend_handles_labels()
    sf.legend(h, l, loc="outside lower center", ncol=2, frameon=False, handlelength=2.4, columnspacing=1.6)
    sf.suptitle("(a) Gate 1 depth, collections under 1M documents", x=0.01, ha="left", fontsize=8)


def panel_b(sf, df):
    ax = sf.subplots(1, 1)
    b = df[df.panel == "b"]
    g = b[b.series != "CRVE setting (star)"]
    for l, (col, mk, ls) in B_STYLE.items():
        p = g[g.chunk_length_tokens == l].sort_values("chunks_per_term")
        ax.plot(p.chunks_per_term, p.gain_ndcg10, color=col, marker=mk, linestyle=ls, label=f"{l} tokens",
                markerfacecolor="white" if l in (64, 128) else col)
    c = b[b.series == "CRVE setting (star)"].iloc[0]
    ax.plot([c.chunks_per_term], [c.gain_ndcg10], marker="*", color=HIGHLIGHT, markersize=8, linestyle="none", label="Confirmed sidecar", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 30, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Chunks kept per term")
    ax.set_ylabel("Gain in nDCG@10")
    ax.set_ylim(0, 0.0120)
    ax.text(0.03, 0.97, "Before the evaluation fixes;\nrandom chunks;\nfive-channel Gate 1", transform=ax.transAxes, va="top", ha="left",
            fontsize=7, linespacing=1.1)
    sf.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, handlelength=2.0, columnspacing=0.7,
              handletextpad=0.35)
    sf.suptitle("(b) Chunk grid at K = 5, over BM25, under 1M", x=0.01, ha="left", fontsize=8)


def panel_c(sf, df):
    import matplotlib.lines as mlines
    axes = sf.subplots(1, 4)
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    c = df[df.panel == "c"]
    order = (("under 1M", "BM25"), ("under 1M", "DPH"), ("over 1M", "BM25"), ("over 1M", "DPH"))
    for ax, (g, base) in zip(axes, order):
        for K, (col, mk, ls) in C_STYLE.items():
            p = c[(c.group == g) & (c.base == base) & (c.K == K)].sort_values("w")
            x = p.w + C_DODGE[K]
            ax.plot(x, p.gain_ndcg10, color=col, marker=mk, linestyle=ls, label=f"K = {K}", markerfacecolor="white" if K == 3 else col,
                    zorder=3)
            for xi, yi, n in zip(x, p.gain_ndcg10, p["collections_below_-0.005"]):
                ax.annotate(f"{int(n)}", (xi, yi), xytext=(0, 5.0 if (K == 5 and abs(xi - 0.2 - C_DODGE[K]) < 1e-9) else 3.2), textcoords="offset points", ha="center", va="bottom", fontsize=7,
                            color="#000000" if n > 0 else "#8a8a8a", zorder=5,
                            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85))
            d = p[p.crve_default.astype(bool)]
            if len(d):
                ax.plot(d.w + C_DODGE[K], d.gain_ndcg10, marker="o", markersize=9, markerfacecolor="none", markeredgecolor=HIGHLIGHT,
                        markeredgewidth=0.9, linestyle="none", zorder=4)
        ax.axhline(0, color="#4d4d4d", linewidth=0.6)
        ax.set_xticks([0.1, 0.2, 0.3])
        ax.set_xticklabels(["0.10", "0.20", "0.30"])
        ax.set_xlim(0.045, 0.355)
        ax.set_xlabel("Weight w")
        ax.set_title(f"{g}, over {base}", fontsize=7.5, pad=2)
    axes[0].set_ylabel("Macro gain in nDCG@10")
    axes[0].set_ylim(0, 0.015)
    axes[2].set_ylim(-0.019, 0.0095)
    for ax in (axes[1], axes[3]):
        ax.tick_params(labelleft=False)
    h, l = axes[0].get_legend_handles_labels()
    h.append(mlines.Line2D([], [], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=HIGHLIGHT, markeredgewidth=0.9,
                           linestyle="none"))
    l.append("Default (K = 5, w = 0.20)")
    sf.legend(h, l, loc="outside lower center", ncol=5, frameon=False, handlelength=2.2, columnspacing=1.4, fontsize=7)
    sf.suptitle("(c) Expansion terms K and weight w; S2, capped used queries, after the fixes, post hoc\n"
                "      Numbers: collections lowered by more than 0.005 (markers offset within each w for legibility)", x=0.01, ha="left",
                fontsize=8, linespacing=1.15)


def draw(df):
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(WIDTH_FULL_CM / 2.54, HEIGHT_CM / 2.54), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.0], hspace=0.02)
    sa, sb = top.subfigures(1, 2, width_ratios=[1.3, 1.0], wspace=0.02)
    for sf in (top, bottom, sa, sb):  # the figure's own white patch is enough; a subfigure patch's corner can print as -0 or 0 between runs
        sf.patch.set_visible(False)
    panel_a(sa, df)
    panel_b(sb, df)
    panel_c(bottom, df)
    path = save_fig(fig, "F_sens")
    plt.close(fig)
    return path


def main():
    f2, f3, t8 = load()
    df = build_data(f2, f3, t8)
    checks(f2, f3, t8, df)
    write("F_sens_data", df, delta_cols=["gain_ndcg10", "ci95_low", "ci95_high"], pct_cols=["gain_rel_pct"],
          int_cols=["L", "K", "chunk_length_tokens", "chunks_per_term", "collections_below_-0.005", "collections"], fixed_cols={"w": 2})
    draw(df)


if __name__ == "__main__":
    main()
