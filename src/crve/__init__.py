"""CRVE: LLM-free query expansion for lexical retrieval from stored corpus contexts.

Subpackages:
  crve.selection  Gate 1 sidecars and proposers, Gate 2 chunking and context sidecar
  crve.scoring    Gate 2 query encoding, chunk scoring and ordering; the queries and index of the live retrieval
  crve.pipeline   the paper's pipeline (Gate 1, Gate 2, S2 sidecar, retrieval, baselines)
"""
