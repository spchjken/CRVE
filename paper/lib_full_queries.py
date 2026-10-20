"""
Corpus-stratified bootstrap contrast, taken from the held-out-queries experiment script (not part of this release).
OUT points at results/full_queries.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import RES

OUT = os.path.join(RES, "full_queries")
SEED = 42


def boot_acc(per: dict, reps: int = 10000, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros(reps)
    for c in sorted(per):
        v = per[c]
        acc += v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return acc / len(per)


def contrast(df: pd.DataFrame, a: str, b: str, metric: str = "ndcg10") -> dict:
    p = df.pivot_table(index=["dataset", "qid"], columns="arm", values=metric, aggfunc="first")
    d = p[a] - p[b]
    per = {c: d.xs(c).to_numpy() for c in sorted(df.dataset.unique())}
    acc = boot_acc(per)
    cm = {c: float(v.mean()) for c, v in per.items()}
    return {"macro": float(np.mean(list(cm.values()))), "pooled": float(d.mean()), "acc": acc, "p": float((acc <= 0).mean()),
            "ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))], "per_corpus": cm,
            "below_floor": int(sum(v < -0.005 for v in cm.values())), "n_corpora": len(cm)}
