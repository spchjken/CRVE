"""The paper's CRVE pipeline and baselines.

The functions in this package compute the paper's numbers: scripts/run_crve.py and the baseline scripts reproduce the
stored per-query outcomes under results/ exactly (README.md). All paths are relative to the repository root, with one
cache layout under data/cache/.

  common     collections, cache paths, the evaluated query lists, Terrier version pin
  assets     Terrier index access, vocabulary pool, restricted (dense) PPMI for collections over 1M documents
  gate1      Gate 1: RRF (k = 60) of the PPMI and sparse lexical profile channels, top L = 200
  gate2      Gate 2: max cosine between the query and each candidate's stored chunks, top K = 5
  sidecar    the S2 context sidecar (n = 20 lowest-IDF-density chunks per pool term)
  retrieval  NoOp term pipeline, live BM25 / DPH retrieval of expanded queries (w = 0.20), per-query metrics
  baselines  RM3 / Bo1 pipelines and tuning grid, BGE-QE / LLM-Q2E replay, per-query metrics from saved runs
"""
