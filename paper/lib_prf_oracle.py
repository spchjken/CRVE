"""
Tuning grid and selection rule of the tuned RM3/Bo1 baselines, taken from the PRF-oracle experiment script (not part
of this release). OUT points at results/prf_oracle.
"""

from __future__ import annotations

import itertools
import os

from common import RES

OUT = os.path.join(RES, "prf_oracle")
DOCS, TERMS, LAMBDAS = (5, 10, 20), (10, 20), (0.3, 0.5, 0.7)
GRID = [("rm3", d, t, l) for d, t, l in itertools.product(DOCS, TERMS, LAMBDAS)] + [("bo1", d, t, None) for d, t in itertools.product(DOCS, TERMS)]
DEFAULT = {"rm3": ("rm3", 10, 10, 0.5), "bo1": ("bo1", 10, 10, None)}


def cfg_name(c):
    return f"{c[0]}_d{c[1]}_t{c[2]}" + (f"_l{c[3]}" if c[3] is not None else "")


def pick(per_q, qids):
    """Best grid point by mean nDCG@10 over qids; ties go to the earlier grid point."""
    m = per_q[per_q.qid.isin(qids)].groupby("config").ndcg10.mean()
    order = [cfg_name(c) for c in GRID if cfg_name(c) in m.index]
    m = m.reindex(order)
    return m.idxmax()
