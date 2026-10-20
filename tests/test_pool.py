"""Pool eligibility and ranking (Section 3.1): DF >= 2, CF >= 3, length >= 2, not all digits, DF/N <= 0.15;
top P by IDF x ln(1 + DF) with the Lucene IDF, ties by term."""
import math

from crve.pipeline.assets import build_pool_v2, lexicon_maps, pool_sha
from evaluation.pool_generators import is_eligible_term


class _Stats:
    def __init__(self, df, cf):
        self.df, self.cf = df, cf

    def getDocumentFrequency(self):
        return self.df

    def getFrequency(self):
        return self.cf


class _Entry:
    def __init__(self, t, df, cf):
        self.t, self.s = t, _Stats(df, cf)

    def getKey(self):
        return self.t

    def getValue(self):
        return self.s


class _Index:
    def __init__(self, rows):
        self.rows = rows

    def getLexicon(self):
        return [_Entry(*r) for r in self.rows]


def test_eligibility_rules():
    n = 100
    assert is_eligible_term("ab", 2, 3, n)
    assert not is_eligible_term("a", 5, 9, n)          # too short
    assert not is_eligible_term("ab", 1, 9, n)         # DF < 2
    assert not is_eligible_term("ab", 5, 2, n)         # CF < 3
    assert not is_eligible_term("1234", 5, 9, n)       # all digits
    assert is_eligible_term("a1", 5, 9, n)             # digits mixed with letters are kept
    assert is_eligible_term("ab", 15, 20, n)           # DF/N = 0.15 is allowed
    assert not is_eligible_term("ab", 16, 20, n)       # DF/N > 0.15


def test_lucene_idf_and_ranking():
    n = 100
    rows = [("alpha", 10, 30), ("beta", 10, 12), ("gamma", 3, 3), ("delta", 50, 80), ("12", 5, 9), ("eps", 1, 5), ("zeta", 10, 11)]
    df, cf, idf = lexicon_maps(_Index(rows), n)
    assert math.isclose(idf["gamma"], math.log(1.0 + (n - 3 + 0.5) / (3 + 0.5)))
    terms, n_el = build_pool_v2(df, cf, idf, n, size=3)
    # eligible: alpha, beta, gamma, zeta (delta: DF/N 0.5; "12": digits; eps: DF 1)
    assert n_el == 4
    sal = {t: idf[t] * math.log(1 + df[t]) for t in ("alpha", "beta", "gamma", "zeta")}
    assert sal["alpha"] == sal["beta"] == sal["zeta"] > sal["gamma"]
    assert terms == ["alpha", "beta", "zeta"]          # equal scores: ties by term
    full, _ = build_pool_v2(df, cf, idf, n, size=10000)
    assert full == ["alpha", "beta", "zeta", "gamma"]  # all eligible terms if fewer than P
    assert pool_sha(["a", "b"]) == pool_sha(["a", "b"]) != pool_sha(["b", "a"])
