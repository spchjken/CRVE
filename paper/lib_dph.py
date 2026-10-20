"""
Readers of the stored BM25 outcomes of the seed-42 draw, taken from the DPH experiment script (not part of this
release). OUT points at results/dph, RED at results/pool_reduced.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import RES
import lib_full_queries as FQ
import lib_pool_size as PS
import lib_weighting as W

OUT = os.path.join(RES, "dph")
RED = os.path.join(RES, "pool_reduced")
SEED = 42
METS = ["ndcg10", "ndcg20", "ndcg50", "recall100", "recall1000"]
BIG = list(W.BIG)


def bm25_side(ds: str) -> pd.DataFrame:
    """BM25 outcomes of the same draws: the seed-42 draw (draw 0) and replicates 1..5 of the full pool."""
    a = pd.read_parquet(os.path.join(PS.OUT, f"outcomes_{ds}.parquet"))
    t = a[(a.order == "salience") & (a.alpha == 1.0)].assign(arm="bm25_exp", draw=0)
    b = a[a.order == "bm25"].assign(arm="bm25", draw=-1)
    r = pd.read_parquet(os.path.join(RED, f"repoutcomes_{ds}.parquet"))
    r = r[r.pool == "full"].rename(columns={"rep": "draw"}).assign(arm="bm25_exp")
    cols = ["dataset", "qid", "arm", "draw"] + METS
    return pd.concat([t[cols], b[cols], r[cols]], ignore_index=True)


def bm25_side_big(ds: str) -> pd.DataFrame:
    f = pd.read_parquet(os.path.join(FQ.OUT, "outcomes.parquet"))
    f = f[(f.dataset == ds) & f.arm.isin(["primary_flat", "bm25"])].replace({"arm": {"primary_flat": "bm25_exp"}})
    g = pd.read_parquet(os.path.join(RES, "gate1_channels", "outcomes.parquet"))
    g = g[(g.dataset == ds) & g.arm.isin(["rrf_ppmi_sparse", "bm25"])].replace({"arm": {"rrf_ppmi_sparse": "bm25_exp"}})
    x = pd.concat([f, g], ignore_index=True)
    x["draw"] = np.where(x.arm == "bm25", -1, 0)
    return x[["dataset", "qid", "arm", "draw"] + METS]
