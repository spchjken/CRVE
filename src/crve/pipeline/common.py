"""Collections, cache layout, evaluated query lists and the Terrier version pin.

All paths are relative to the repository root; run every script from there (`python scripts/<x>.py`; the scripts put src/
on the path themselves).

Cache layout (data/cache/, ignored by git):
  terrier_indices/<c>_default/          Terrier index (scripts/build_index.py; settings in configs/index_settings.yaml)
  gate1/<c>_canonical_pool.json         vocabulary pool (scripts/build_assets.py pool)
  gate1/<c>_bounded_ppmi.parquet        PPMI co-occurrence lists (build_assets.py ppmi; over 1M restricted to the
                                        anchors of the evaluated queries)
  gate1/<c>_lexical_profiles.parquet    sparse lexical profiles and their Terrier index (build_assets.py profiles,
  gate1/<c>_lexical_profiles_idx/       or the stored profiles of the paper)
  sidecars/S2/<c>/                      S2 context sidecar (scripts/build_sidecar.py, or the stored sidecars)
  qe_llm/raw/                           LLM-Q2E keyword cache (stored generations, replayed)
  qe_vocab/                             BGE-QE vocabulary sidecar (built on first use)
"""

from __future__ import annotations

import os

# the 20 collections under one million documents (sorted) and the five over
UNDER_1M = sorted(["bright_aops", "bright_economics", "bright_leetcode", "bright_stackoverflow", "bright_sustainable_living", "fiqa",
                   "nfcorpus", "quora", "scifact", "arguana", "bright_biology", "bright_earth_science", "bright_pony",
                   "bright_psychology", "bright_robotics", "bright_theoremqa_questions", "bright_theoremqa_theorems", "scidocs",
                   "trec_covid", "webis_touche2020"])
OVER_1M = ["nq", "dbpedia_entity", "hotpotqa", "fever", "climate_fever"]

GATE1_DIR = "data/cache/gate1"
SIDECAR_DIR = "data/cache/sidecars/S2"
QUERIES_DIR = "results/queries"

TERRIER_VERSION = "5.11"
TERRIER_HELPER_VERSION = "0.0.8"

# per-query metrics stored by the runs (linear-gain nDCG and recall at depth 1,000, after exclusions)
METRICS = ["ndcg10", "ndcg20", "ndcg30", "ndcg50", "recall50", "recall100", "recall200", "recall500", "recall1000"]


def pin_terrier() -> None:
    """Pins the Terrier release used for the paper (5.11). Call before anything starts the JVM."""
    import pyterrier as pt
    if not pt.java.started():
        pt.terrier.set_version(TERRIER_VERSION)
        pt.terrier.set_helper_version(TERRIER_HELPER_VERSION)


def queries(ds: str) -> list:
    """The evaluated query ids of a collection (results/queries/<c>.txt), sorted."""
    with open(os.path.join(QUERIES_DIR, f"{ds}.txt"), encoding="utf-8") as f:
        return sorted(line.rstrip("\n") for line in f if line.strip())


def pool_file(ds: str) -> str:
    return os.path.join(GATE1_DIR, f"{ds}_canonical_pool.json")


def sidecar_dir(ds: str, root: str = SIDECAR_DIR) -> str:
    return os.path.join(root, ds)
