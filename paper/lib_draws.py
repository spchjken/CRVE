"""
Reader of the seed-42 draw outcomes, taken from the six-draw experiment script (not part of this release).
OUT points at results/draws.
"""

from __future__ import annotations

import os

import pandas as pd

from common import RES
import lib_dph as D
import lib_pool_size as PS
import lib_weighting as W

OUT = os.path.join(RES, "draws")
SMALL = list(PS.CORPORA)
BIG = list(W.BIG)
METS = D.METS
K = 5


def seed42_outcomes(ds):
    """The seed-42 draw: BM25 and DPH outcomes."""
    d = pd.read_parquet(os.path.join(D.OUT, f"outcomes_{ds}.parquet"))
    d = d[(d.arm == "dph_exp") & (d.draw == 0)].assign(model="dph")
    b = D.bm25_side_big(ds) if ds in BIG else D.bm25_side(ds)
    b = b[(b.arm == "bm25_exp") & (b.draw == 0)].assign(model="bm25")
    return pd.concat([d, b], ignore_index=True).assign(arm="seed42", seed=42)[["dataset", "qid", "model", "arm", "seed"] + METS]
