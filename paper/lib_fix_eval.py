"""
Constants taken from the evaluation-fix rerun script (not part of this release); OUT points at results/fix_eval.
The four collections whose outcomes were recomputed after the evaluation fixes.
"""

from __future__ import annotations

import os

from common import RES
import lib_dph as D

OUT = os.path.join(RES, "fix_eval")
FIXED = ["arguana", "bright_biology", "bright_earth_science", "bright_stackoverflow"]
METS = D.METS
