#!/usr/bin/env python3
"""Build the Terrier index of a collection with the settings of the paper's runs (configs/index_settings.yaml).

  python scripts/build_index.py --corpus scifact
  python scripts/build_index.py --corpus scifact --check-only        # check an existing index against the frozen statistics

Writes data/cache/terrier_indices/<c>_default (or --out-dir). Terrier 5.11 (pinned), English tokeniser, stopword list and
Porter stemmer; the document id length, the stored-text length and the maximum term length are per collection. Documents
are streamed from data/raw (crve: evaluation.benchmark_loader.BenchmarkLoader.stream_corpus); indexing memory 1 GiB, JVM
heap 4 GiB. After the build, the collection statistics and the lexicon hash are compared with the frozen values; any
difference stops with exit status 1.

Indexing code: the indexer calls of the scripts that built the paper's indexes (IterDictIndexer with meta lengths,
max.term.length and indexing.max.memory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import yaml

SETTINGS = "configs/index_settings.yaml"


def props(path):
    out = {}
    for line in open(os.path.join(path, "data.properties"), encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            k, v = line.rstrip("\n").split("=", 1)
            out[k] = v
    return out


def check(pt, path, s):
    """Statistics and lexicon hash of the index at `path` against the frozen settings `s`."""
    from crve.selection.gate1_sidecars import compute_lexicon_semantic_hash
    p = props(path)
    got = {"docno_length": int(p["index.meta.value-lengths"].split(",")[0]), "text_length": int(p["index.meta.value-lengths"].split(",")[1]),
           "max_term_length": int(p["max.term.length"]), "termpipelines": p["termpipelines"],
           "num_documents": int(p["num.Documents"]), "num_terms": int(p["num.Terms"]), "num_tokens": int(p["num.Tokens"]),
           "num_pointers": int(p["num.Pointers"]), "terrier_version": p.get("index.terrier.version")}
    idx = pt.IndexFactory.of(os.path.join(os.path.abspath(path), "data.properties"))
    got["lexicon_sha256"] = compute_lexicon_semantic_hash(idx)
    want = dict(s, terrier_version="5.11")
    diff = {k: (got[k], want[k]) for k in want if got.get(k) != want[k]}
    return got, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", default="data/cache/terrier_indices")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    s = yaml.safe_load(open(SETTINGS, encoding="utf-8"))["collections"][a.corpus]
    from crve.pipeline.common import pin_terrier
    pin_terrier()
    import pyterrier as pt
    if not pt.java.started():
        pt.java.set_memory_limit(4096)
        pt.java.init()
    dst = os.path.abspath(os.path.join(a.out_dir, f"{a.corpus}_default"))
    if not a.check_only:
        if os.path.exists(os.path.join(dst, "data.properties")) and not a.overwrite:
            raise SystemExit(f"{dst} exists (use --overwrite to rebuild, --check-only to check it)")
        from evaluation.benchmark_loader import BenchmarkLoader
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        indexer = pt.IterDictIndexer(dst, overwrite=True, meta={"docno": int(s["docno_length"]), "text": int(s["text_length"])})
        indexer.setProperty("max.term.length", str(s["max_term_length"]))
        indexer.setProperty("termpipelines", s["termpipelines"])
        indexer.setProperty("indexing.max.memory", "1073741824")
        indexer.index({"docno": str(d), "text": t} for d, t in BenchmarkLoader.stream_corpus(a.corpus))
        print(json.dumps({"corpus": a.corpus, "built": dst, "seconds": round(time.time() - t0, 1)}), flush=True)
    got, diff = check(pt, dst, s)
    print(json.dumps({"corpus": a.corpus, "index": dst, "statistics": got, "differences": diff}, indent=1), flush=True)
    if diff:
        raise SystemExit(f"{a.corpus}: index differs from the frozen settings: {diff}")


if __name__ == "__main__":
    main()
