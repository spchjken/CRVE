"""
Gate 2 ordering of the candidates (`ranked_list`): score descending (candidates without a score last), then Gate 1 rank,
then term. Label-free: only candidate scores and Gate 1 ranks.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def ranked_list(terms: Sequence[str], scores: Sequence[float], ext_rank: Sequence[float], top: int = 10) -> List[str]:
    """Score descending (NaN last), then Gate 1 rank ascending, then term ascending."""
    s = np.asarray(scores, dtype=float)
    r = np.asarray(ext_rank, dtype=float)
    key = [(0 if np.isfinite(si) else 1, -si if np.isfinite(si) else 0.0, ri if np.isfinite(ri) else np.inf, t)
           for t, si, ri in zip(terms, s, r)]
    order = sorted(range(len(key)), key=lambda i: key[i])
    return [terms[i] for i in order[:top]]
