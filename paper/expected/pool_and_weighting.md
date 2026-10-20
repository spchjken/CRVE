# Pool size and per-term weighting (Section 5.3)

Written by `paper/s52_53.py`.

**Pool size.** A pool of 30% of the eligible terms, rebuilt in full (five builds, against five builds of the full pool), keeps 85% of the gain in nDCG@10 over BM25 (ratio of the mean gains of the five builds; query-bootstrap interval 66% to 103%); FiQA, Sustainable Living and ArguAna keep only 36% to 49% of their own gains. Measured before the evaluation corrections (ArguAna included), with randomly drawn chunks (seed 42), the two-channel Gate 1, K = 5 and w = 0.20. Sources: `results/pool_reduced/rep_verdict.json`, `rep_extra.json`.

**Per-term weighting.** Weighting each term by the certainty of its chunk sample, against the flat weight w = 0.20 (randomly drawn chunks, the 30,541 queries of Section 5.3): under 1M +0.0005, Holm-adjusted 97.5% interval [−0.0031, +0.0047]; over 1M −0.0008, 95% interval [−0.0021, +0.0004]. Source: `results/recheck/full_queries/verdict.json`.
