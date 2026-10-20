"""The S2 context sidecar: per pool term, the n = 20 chunks with the lowest IDF density among all of the term's chunks.

  chunks_of   one pass over the corpus: every chunk that contains a pool term, with its matched pool terms and stems
  keep        keeps the n largest keys of a term (key: -IDF density, then -corpus order)
  build_s2    fills the per-term chunk lists and writes the sidecar with StreamingSidecarBuilder.finalize_sidecar (chunk
              texts, term references, BGE-small-en-v1.5 CLS vectors encoded in float16 on the GPU)

Chunks are cut by crve.selection.gate2_contracts.segment_document_canonical: about 100 sub-word tokens (boundary
within 85 to 115, preferring sentence ends), at most 115. A chunk is a context of every pool term that occurs in it after
Terrier analysis. IDF density = mean Lucene-style IDF of the chunk's analysed tokens (0 for tokens not in the lexicon).
Ties go to the earlier chunk in corpus order.
"""

from __future__ import annotations

import heapq
import json
import os
import shutil
import time

import numpy as np

CAP = 20
DOCS = {"n": 0}


def chunks_of(builder, ds):
    """Yields (order, SourceChunkRef, matched pool terms, analysed stems) over the corpus."""
    from crve.selection.gate2_contracts import SourceChunkRef, segment_document_canonical
    from evaluation.benchmark_loader import BenchmarkLoader
    k = 0
    DOCS["n"] = 0
    for doc_id, text in BenchmarkLoader.stream_corpus(ds):
        DOCS["n"] += 1
        if not text or not text.strip():
            continue
        for ch in segment_document_canonical(doc_id=doc_id, doc_text=text, tokenizer=builder.tokenizer,
                                             target_subwords=builder.target_subwords, max_subwords=builder.max_subwords,
                                             snap_window=builder.snap_window):
            stems, _ = builder.analyzer.analyze(ch.text)
            matched = set(stems) & builder.pool_set
            if not matched:
                continue
            ref = SourceChunkRef(chunk_id=ch.chunk_id, doc_id=ch.doc_id, chunk_idx=ch.chunk_idx, char_start=ch.char_start,
                                 char_end=ch.char_end, tok_count=ch.tok_count, text=ch.text,
                                 is_overlong_compound=ch.is_overlong_compound)
            yield k, ref, matched, stems
            k += 1


def keep(heap, key, item):
    """Keeps the CAP largest keys; the root is the current worst."""
    if len(heap) < CAP:
        heapq.heappush(heap, (key, item))
    elif key > heap[0][0]:
        heapq.heapreplace(heap, (key, item))


def build_s2(ds, pool_path, out_root):
    """Builds out_root/<ds>/ (chunk_embeddings.npy, chunks.parquet, term_references.parquet, sidecar_manifest.json)."""
    from crve.pipeline import assets as B
    from crve.pipeline import gate1 as G
    from crve.selection.gate2_sidecar import StreamingSidecarBuilder
    t0 = time.time()
    os.makedirs(out_root, exist_ok=True)
    staging = os.path.join(out_root, f".tmp_{ds}_{os.getpid()}")
    b = StreamingSidecarBuilder(dataset=ds, pool_path=pool_path, output_dir=staging,
                                compute_dtype="float16", seed=0, batch_size=64, target_subwords=100,
                                max_subwords=115, snap_window=15, capacity_b=CAP)
    pt = B.init_terrier(G.g1_with_seed(42))
    idx, n_docs = B.open_index(pt, ds)
    _, _, idf = B.lexicon_maps(idx, n_docs)
    pop = {t: 0 for t in b.pool_set}
    h2 = {t: [] for t in b.pool_set}
    for k, ref, matched, stems in chunks_of(b, ds):
        idfd = float(np.mean([idf.get(s, 0.0) for s in stems]))
        for t in matched:
            pop[t] += 1
            keep(h2[t], (-idfd, -k), ref)
    docs = DOCS["n"]
    t1 = time.time()
    print(json.dumps({"corpus": ds, "pass": 1, "seconds": round(t1 - t0, 1)}), flush=True)
    for t in b.pool_set:
        res = b.reservoirs[t]
        res.population_N = pop[t]
        res.sample = [r for _, r in sorted(h2[t], key=lambda x: x[0], reverse=True)]
        res.distinct_docs = {r.doc_id for r in res.sample}
    b.documents_processed, b.max_docs, b.corpus_completed = docs, None, True
    b.ingestion_time_seconds = t1 - t0
    man = b.finalize_sidecar()
    dst = os.path.join(out_root, ds)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(staging, dst)
    stats = {k_: man.get(k_) for k_ in ("unique_chunks_encoded", "total_sampled_chunk_references", "encoding_time_seconds")}
    with open(os.path.join(out_root, f"build_{ds}.json"), "w", encoding="utf-8") as f:
        json.dump({"corpus": ds, "seconds": round(time.time() - t0, 1), "pass1_s": round(t1 - t0, 1), "S2": stats}, f, indent=2)
    print(json.dumps({"corpus": ds, "seconds": round(time.time() - t0, 1), "S2": stats}), flush=True)
    return man
