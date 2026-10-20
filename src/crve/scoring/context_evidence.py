"""
Gate 2 query encoding and chunk similarities.

The query is encoded with BAAI/bge-small-en-v1.5 at a pinned revision (CPU, float32, CLS pooling, L2 normalisation, at
most 128 tokens, no instruction prefix), and compared with every stored chunk of every candidate term; the scaled score
(cos + 1) / 2, clipped to [0, 1], orders like the cosine.

Query texts are loaded without labels (`BenchmarkLoader.load_query_texts_only`); no qrels or retrieval are used here.
"""

from __future__ import annotations

import gc
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import numpy as np

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBED_DIM = 384


def encode_queries(
    query_texts: Mapping[Tuple[str, str], str],
    device: str = "cpu",
    seed: int = 42,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Encodes (dataset, qid) -> text with the Gate 2 conventions above. Returns unit vectors."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModel.from_pretrained(BGE_MODEL_NAME, revision=BGE_REVISION, dtype=torch.float32).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME, revision=BGE_REVISION)
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for key in sorted(query_texts):
        inputs = tokenizer(query_texts[key], return_tensors="pt", truncation=True, max_length=128, padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, 0, :]
            vec = torch.nn.functional.normalize(hidden, p=2, dim=1)[0].cpu().to(torch.float32).numpy()
        if vec.shape != (EMBED_DIM,) or not np.isfinite(vec).all():
            raise ValueError(f"invalid query vector for {key}")
        out[key] = vec
    del model, tokenizer
    gc.collect()
    return out


def score_query_terms(
    sidecar,
    query_vec: np.ndarray,
    terms: Iterable[str],
) -> Tuple[List[Dict], np.ndarray]:
    """
    Scores every retained chunk of every term against one query.
    Returns (rows, pool) where rows hold per-term metadata and the scaled similarity list (empty for
    terms absent from the sidecar or unverified), and pool is the array of scaled similarities over the
    union of unique chunks referenced by these terms (the query's candidate chunk pool).
    """
    metas = {}
    union: Set[int] = set()
    for t in terms:
        m = sidecar.get_term_metadata(t)
        metas[t] = m
        if m is not None and m["status"] == "VERIFIED" and m["sample_n"] > 0:
            union.update(int(i) for i in m["chunk_indices"])
    idx = np.array(sorted(union), dtype=np.int64)
    sims: Dict[int, float] = {}
    pool = np.empty(0, dtype=np.float64)
    if len(idx):
        emb = np.asarray(sidecar.embeddings[idx], dtype=np.float32)
        raw = np.clip(emb @ query_vec.astype(np.float32), -1.0, 1.0)
        scaled = np.clip((raw.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
        sims = dict(zip(idx.tolist(), scaled.tolist()))
        pool = scaled
    rows = []
    for t, m in metas.items():
        if m is None:
            rows.append({"term": t, "in_sidecar": False, "verified": False, "sample_n": 0, "population_n": 0,
                         "distinct_docs": 0, "sims": []})
            continue
        ok = m["status"] == "VERIFIED" and m["sample_n"] > 0
        rows.append({
            "term": t, "in_sidecar": True, "verified": ok,
            "sample_n": int(m["sample_n"]), "population_n": int(m["population_N"]),
            "distinct_docs": int(m["distinct_docs_S"]),
            "sims": [sims[int(i)] for i in m["chunk_indices"]] if ok else [],
        })
    return rows, pool
