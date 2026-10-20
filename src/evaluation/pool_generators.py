"""
The vocabulary pool rule of CRVE (Section 3.2 of the paper): a term is eligible if it occurs in at least two documents and at
least three times, has at least two characters, is not made of digits only and occurs in at most 15% of the documents
(`is_eligible_term`); eligible terms are ranked by IDF(t) x ln(1 + DF(t)) (`compute_salience_score`) and the top P form the
pool (crve.pipeline.assets.build_pool_v2).
"""

import math


def is_eligible_term(term: str, df: int, cf: int, num_docs: int) -> bool:
    """
    Operational eligibility reliability filter:
    DF >= 2, CF >= 3, len >= 2, not pure digits, and DF/N <= 0.15.
    """
    if len(term) < 2:
        return False
    if df < 2 or cf < 3:
        return False
    if (df / max(num_docs, 1)) > 0.15:
        return False
    if term.isdigit():
        return False
    return True


def compute_salience_score(idf: float, df: int) -> float:
    """Score_sal(t) = IDF(t) * ln(1 + DF(t))"""
    return idf * math.log(1.0 + max(df, 0))
