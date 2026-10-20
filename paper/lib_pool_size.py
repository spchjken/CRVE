"""
Constants of the pool-size experiment (not part of this release); OUT points at results/pool_size.
"""

from __future__ import annotations

import os

from common import RES

OUT = os.path.join(RES, "pool_size")
# the 20 collections under one million documents, sorted
CORPORA = sorted(["arguana", "bright_aops", "bright_biology", "bright_earth_science", "bright_economics", "bright_leetcode",
                  "bright_pony", "bright_psychology", "bright_robotics", "bright_stackoverflow", "bright_sustainable_living",
                  "bright_theoremqa_questions", "bright_theoremqa_theorems", "fiqa", "nfcorpus", "quora", "scidocs", "scifact",
                  "trec_covid", "webis_touche2020"])
