#!/usr/bin/env python3
"""Build the S2 context sidecar of a collection (Gate 2): per pool term, its n = 20 chunks with the lowest IDF density.

  python scripts/build_sidecar.py --corpus scifact

Needs the Terrier index, the pool (data/cache/gate1/<c>_canonical_pool.json, or --pool-file) and a CUDA GPU (the chunk
vectors are encoded in float16, as for the paper). Writes data/cache/sidecars/S2/<c>/ (or --out-root/<c>/):
chunks.parquet (unique chunk texts), chunk_embeddings.npy (float16, 384 dimensions), term_references.parquet (per term:
population, stored chunks in S2 order, their documents and character offsets), sidecar_manifest.json (checksums).

Chunks: about 100 BGE sub-word tokens, boundary within +-15 tokens of the target (sentence ends first), at most 115
(crve.selection.gate2_contracts.segment_document_canonical). Encoder: BAAI/bge-small-en-v1.5, revision
5c38ec7c405ec4b44b94cc5a9bb96e735b38267a, CLS pooling, unit length. Implementation: crve.pipeline.sidecar.build_s2.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pool-file", default=None)
    ap.add_argument("--out-root", default="data/cache/sidecars/S2")
    a = ap.parse_args()
    from crve.pipeline import common as C
    from crve.pipeline.sidecar import build_s2
    C.pin_terrier()
    build_s2(a.corpus, a.pool_file or C.pool_file(a.corpus), a.out_root)


if __name__ == "__main__":
    main()
