"""
Reader of the six random draws and the base per query, taken from the chunk-selection (S1/S2) experiment script
(not part of this release). OUT points at results/feature_sidecar.
"""

from __future__ import annotations

import os

import pandas as pd

from common import RES
import lib_direct as RD
import lib_draws as R
import lib_fix_eval as FX

OUT = os.path.join(RES, "feature_sidecar")
SMALL = RD.SMALL
BIG = RD.BIG
METS = R.METS


def reference(ds):
    """Six random draws (seeds 1-5 and seed 42) and the base, per query and model."""
    if ds in FX.FIXED:
        o = pd.read_parquet(os.path.join(FX.OUT, f"outcomes_{ds}.parquet")).replace({"arm": {"today": "seed42"}})  # stored label
    else:
        o = pd.concat([pd.read_parquet(os.path.join(R.OUT, f"outcomes_{ds}.parquet")), R.seed42_outcomes(ds)], ignore_index=True)
    return o
