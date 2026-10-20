"""
Shared conventions for the paper's tables and figures.
Read-only re-aggregation of stored result files under results/; no retrieval.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "paper") not in sys.path:  # the paper scripts import each other (common, lib_*); nothing from src/
    sys.path.insert(0, os.path.join(ROOT, "paper"))

import numpy as np
import pandas as pd

RES = os.path.join(ROOT, "results")  # shipped result files (inputs)
OUT = os.path.join(ROOT, "paper", "output")  # regenerated tables
FIG = os.path.join(OUT, "figures")  # regenerated figures

UNDER = ["scifact", "nfcorpus", "trec_covid", "fiqa", "quora", "arguana", "scidocs", "webis_touche2020", "bright_aops", "bright_economics",
         "bright_leetcode", "bright_stackoverflow", "bright_sustainable_living", "bright_biology", "bright_earth_science", "bright_psychology",
         "bright_robotics", "bright_pony", "bright_theoremqa_questions", "bright_theoremqa_theorems"]
OVER = ["nq", "dbpedia_entity", "hotpotqa", "fever", "climate_fever"]
NAME = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "trec_covid": "TREC-COVID", "fiqa": "FiQA", "quora": "Quora", "arguana": "ArguAna",
        "scidocs": "SCIDOCS", "webis_touche2020": "Touché-2020", "bright_aops": "AoPS", "bright_economics": "Economics",
        "bright_leetcode": "LeetCode", "bright_stackoverflow": "Stack Overflow", "bright_sustainable_living": "Sustainable Living",
        "bright_biology": "Biology", "bright_earth_science": "Earth Science", "bright_psychology": "Psychology", "bright_robotics": "Robotics",
        "bright_pony": "Pony", "bright_theoremqa_questions": "TheoremQA-Questions", "bright_theoremqa_theorems": "TheoremQA-Theorems",
        "nq": "NQ", "dbpedia_entity": "DBPedia-Entity", "hotpotqa": "HotpotQA", "fever": "FEVER", "climate_fever": "Climate-FEVER"}
GROUP_NAME = {"under_1M": "under 1M", "over_1M": "over 1M"}


def group_of(ds):
    return "over_1M" if ds in OVER else "under_1M"


def rel(diff, ref):
    """Relative change in percent of diff against a reference value; None if the reference is zero or missing."""
    if ref is None or not np.isfinite(ref) or ref == 0:
        return np.nan
    return 100.0 * diff / ref


def lvl(x):
    """Interval level as a percent string: 0.9875 -> '98.75%', 0.95 -> '95%'."""
    s = f"{100 * float(x):.2f}".rstrip("0").rstrip(".")
    return f"{s}%"


def fmt_fixed(x, nd):
    """Fixed-decimal string with trailing zeros kept; values that round to zero print without a sign; missing values print empty."""
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)) or pd.isna(x):
        return ""
    s = f"{float(x):.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return s


def round_table(df, delta_cols=(), level_cols=(), pct_cols=(), int_cols=(), level4_cols=(), holm_level_cols=(), fixed_cols=None):
    """As printed, as fixed-decimal strings: 4 decimals for nDCG deltas and intervals (and level4_cols), 3 for levels, 1 for
    relative percent; holm_level_cols as percent strings ('98.75%'); fixed_cols {column: decimals} for parameters such as w.
    Every float column must be listed (else SystemExit)."""
    r = df.copy()
    done = set()
    groups = [(delta_cols, 4), (level4_cols, 4), (level_cols, 3), (pct_cols, 1)] + [((c,), nd) for c, nd in (fixed_cols or {}).items()]
    for cols, nd in groups:
        for c in cols:
            if c in r:
                r[c] = [fmt_fixed(v, nd) for v in r[c].astype(float)]
                done.add(c)
    for c in holm_level_cols:
        if c in r:
            r[c] = ["" if pd.isna(v) else lvl(v) for v in r[c]]
            done.add(c)
    for c in int_cols:
        if c in r:
            r[c] = r[c].astype("Int64")
            done.add(c)
    left = [c for c in r.columns if c not in done and pd.api.types.is_float_dtype(r[c])]
    if left:
        raise SystemExit(f"round_table: float columns without a rounding rule: {left}")
    return r


def write(table_id, df, **round_kw):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{table_id}.csv"), index=False)
    round_table(df, **round_kw).to_csv(os.path.join(OUT, f"{table_id}_rounded.csv"), index=False)
    print(f"{table_id}: {len(df)} rows", flush=True)


def boot_ci(x, reps=10000, seed=42):
    """95 percent percentile interval of the mean of x (queries resampled with replacement)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---- figure style: one palette, readable in greyscale, no titles, plain-English labels ----
WIDTH_FULL_CM = 17.0
WIDTH_12_CM, WIDTH_6_CM = 12.0, 6.0
MARKERS = {"bm25": "o", "dph": "s"}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "font.family": "serif", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.5, "lines.linewidth": 1.0, "lines.markersize": 3.5})
    return plt


def new_fig(width_cm, height_cm, nrows=1, ncols=1, **kw):
    """Figure of exactly width_cm x height_cm (saved without bbox_inches='tight'), laid out by the constrained layout engine."""
    plt = setup_matplotlib()
    return plt.subplots(nrows, ncols, figsize=(width_cm / 2.54, height_cm / 2.54), layout="constrained", **kw)


def save_fig(fig, fig_id):
    """Save at the figure's own size (full width 17 cm, or the 12 / 6 cm widths of figures placed inside a text column); no CreationDate, so the PDF is byte-reproducible."""
    w_cm = fig.get_figwidth() * 2.54
    if not any(abs(w_cm - w) < 1e-9 for w in (WIDTH_FULL_CM, WIDTH_12_CM, WIDTH_6_CM)):
        raise SystemExit(f"{fig_id}: figure width {w_cm:.3f} cm is none of {WIDTH_FULL_CM}, {WIDTH_12_CM}, {WIDTH_6_CM} cm")
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, f"{fig_id}.pdf")
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    print(f"{fig_id}: {os.path.relpath(path, ROOT)} ({w_cm:.1f} cm x {fig.get_figheight() * 2.54:.1f} cm)", flush=True)
    return path
