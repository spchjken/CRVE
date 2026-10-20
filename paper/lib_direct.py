"""
Collection lists and the two-sided Holm step-down, taken from an experiment script (not part of this release).
"""

from __future__ import annotations

import numpy as np

import lib_pool_size as PS
import lib_weighting as W

SMALL = list(PS.CORPORA)
BIG = list(W.BIG)


def holm2(per_contrast):
    """Two-sided Holm step-down over bootstrap replicates; decided only while every earlier step decided."""
    p = {k: 2 * min((v["acc"] <= 0).mean(), (v["acc"] >= 0).mean()) for k, v in per_contrast.items()}
    order = sorted(p, key=lambda k: (p[k], k))
    m, going, out = len(order), True, {}
    for i, k in enumerate(order):
        alpha = 0.05 / (m - i)
        lo, hi = np.percentile(per_contrast[k]["acc"], [100 * alpha / 2, 100 - 100 * alpha / 2])
        dec = going and (lo > 0 or hi < 0)
        going = dec
        out[k] = {"holm_step": i + 1, "level": round(1 - alpha, 4), "interval": [float(lo), float(hi)], "p_two_sided": float(p[k]),
                  "decided": bool(dec), "direction": ("above 0" if lo > 0 else "below 0") if dec else "undecided"}
    return out
