"""Gate 2 context sidecar: the builder's finalisation (chunk texts, term references, encoder vectors) and the runtime
container (ContextSidecar).

The chunks kept per term are chosen by crve.pipeline.sidecar (S2: the n = 20 chunks of lowest IDF density); this module
deduplicates their texts, encodes them with BAAI/bge-small-en-v1.5 at a pinned revision (CLS pooling, unit length,
float16 on the GPU), writes chunks.parquet, chunk_embeddings.npy, term_references.parquet and sidecar_manifest.json, and
loads them back with checksum verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from crve.selection.gate2_contracts import (
    DEFAULT_RESERVOIR_CAPACITY,
    MAX_CHUNK_SUBWORDS,
    MAX_ENCODER_TOKENS,
    SNAP_WINDOW,
    TARGET_CHUNK_SUBWORDS,
    SourceChunkRef,
)

# Immutable Dense Model Constants & Schema
BGE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
BGE_REVISION: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_EMBEDDING_DIM: int = 384
SIDECAR_SCHEMA_VERSION: str = "2.0.0"
DEFAULT_ANALYZER_CONFIG: str = "pyterrier_terrier_stemmer"


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes the SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_corpus_sha256(dataset: str) -> str:
    """Computes a deterministic SHA256 checksum over the raw corpus files consumed by stream_corpus()."""
    from evaluation.benchmark_loader import BenchmarkLoader
    paths = BenchmarkLoader.get_corpus_source_paths(dataset)
    hasher = hashlib.sha256()
    for p in sorted(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Raw corpus file not found: {p}")
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def derive_term_seed(global_seed: int, term: str) -> int:
    """Derives a deterministic 32-bit integer seed for a candidate term.

    Guarantees that a term's reservoir sample is an invariant function of its
    own occurrence sequence, remaining strictly independent of other terms in the pool.
    """
    key = f"{global_seed}:{term}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


@dataclass
class TermReservoir:
    """The chunks kept for one pool term: sample (at most `capacity`), population_N = number of chunks containing the
    term, and the distinct documents of the sample."""

    term: str
    capacity: int = DEFAULT_RESERVOIR_CAPACITY
    population_N: int = 0
    sample: List[SourceChunkRef] = field(default_factory=list)
    distinct_docs: Set[str] = field(default_factory=set)
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    @property
    def sample_n(self) -> int:
        return len(self.sample)

    @property
    def distinct_docs_S(self) -> int:
        return len(self.distinct_docs)

    @property
    def status(self) -> str:
        return "VERIFIED" if self.sample_n > 0 else "UNVERIFIED"


class StreamingSidecarBuilder:
    """Holds the chunks kept per pool term (filled by crve.pipeline.sidecar.build_s2) and writes the Gate 2 context sidecar.

    - Chunk texts are deduplicated into one (N_unique, 384) float16 matrix; the source identity of every kept chunk
      (chunk_id, doc_id, chunk_idx, char_start, char_end) is kept in term_references.parquet.
    - Terms without any chunk get status UNVERIFIED and empty chunk lists.
    """

    def __init__(
        self,
        dataset: str,
        pool_path: str,
        output_dir: str,
        corpus_id: Optional[str] = None,
        corpus_sha256: Optional[str] = None,
        analyzer_config: Optional[str] = None,
        compute_dtype: Optional[str] = None,
        seed: int = 42,
        batch_size: int = 64,
        target_subwords: int = TARGET_CHUNK_SUBWORDS,
        max_subwords: int = MAX_CHUNK_SUBWORDS,
        snap_window: int = SNAP_WINDOW,
        capacity_b: int = DEFAULT_RESERVOIR_CAPACITY,
        model_name: str = BGE_MODEL_NAME,
        revision: str = BGE_REVISION,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        analyzer: Optional[Any] = None,
    ):
        self.dataset = dataset
        self.pool_path = pool_path
        self.output_dir = output_dir
        self.corpus_id = corpus_id or dataset
        self.analyzer_config = analyzer_config or DEFAULT_ANALYZER_CONFIG
        self.schema_version = SIDECAR_SCHEMA_VERSION
        self.seed = seed
        self.batch_size = batch_size
        self.target_subwords = target_subwords
        self.max_subwords = max_subwords
        self.snap_window = snap_window
        self.capacity_b = capacity_b
        self.model_name = model_name
        self.revision = revision

        # Strict upfront compute_dtype resolution
        if compute_dtype is not None:
            self.compute_dtype = compute_dtype
        else:
            is_cpu_model = False
            if model is not None:
                try:
                    p = next(model.parameters(), None)
                    if p is not None and p.device.type == "cpu":
                        is_cpu_model = True
                    elif p is None:
                        is_cpu_model = True
                except Exception:
                    is_cpu_model = True
            self.compute_dtype = "float32" if (is_cpu_model or not torch.cuda.is_available()) else "float16"

        if self.compute_dtype == "float16" and not torch.cuda.is_available():
            raise ValueError("compute_dtype='float16' requires CUDA, but CUDA is not available.")
        if self.compute_dtype not in ["float16", "float32"]:
            raise ValueError(f"Unsupported compute_dtype: {self.compute_dtype}")

        # Resolve corpus content hash
        if corpus_sha256 is not None:
            self.corpus_sha256 = corpus_sha256
        else:
            # Strict: Fail loudly if raw corpus cannot be hashed; no silent synthetic fallback
            self.corpus_sha256 = compute_corpus_sha256(self.dataset)

        # Strict parameter validation
        if self.capacity_b <= 0:
            raise ValueError(f"capacity_b must be positive, got {self.capacity_b}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.target_subwords <= 0:
            raise ValueError(f"target_subwords must be positive, got {self.target_subwords}")
        if self.snap_window < 0:
            raise ValueError(f"snap_window must be non-negative, got {self.snap_window}")
        if self.max_subwords > MAX_CHUNK_SUBWORDS:
            raise ValueError(
                f"max_subwords ({self.max_subwords}) exceeds pilot specification limit "
                f"MAX_CHUNK_SUBWORDS ({MAX_CHUNK_SUBWORDS})."
            )
        if self.max_subwords + 2 > MAX_ENCODER_TOKENS:
            raise ValueError(
                f"max_subwords ({self.max_subwords}) + 2 special tokens exceeds "
                f"MAX_ENCODER_TOKENS ({MAX_ENCODER_TOKENS}). Silent truncation is prohibited."
            )

        # Compute provenance hashes for cache verification
        with open(self.pool_path, "rb") as f:
            self.pool_sha256: str = hashlib.sha256(f.read()).hexdigest()

        config_str = (
            f"schema:{self.schema_version}:dataset:{self.dataset}:corpus:{self.corpus_id}:"
            f"corpus_sha256:{self.corpus_sha256}:pool:{self.pool_sha256}:"
            f"target:{self.target_subwords}:max:{self.max_subwords}:"
            f"snap:{self.snap_window}:b:{self.capacity_b}:seed:{self.seed}:"
            f"model:{self.model_name}:{self.revision}:dtype:{self.compute_dtype}:"
            f"analyzer:{self.analyzer_config}"
        )
        self.config_hash: str = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

        # Build lifecycle & timing state
        self.documents_processed: int = 0
        self.max_docs: Optional[int] = None
        self.ingestion_time_seconds: float = 0.0
        self.corpus_completed: bool = False

        # Load canonical pool
        self.pool_terms: List[str] = self._load_canonical_pool(pool_path)
        self.pool_set: Set[str] = set(self.pool_terms)

        # Initialize independent reservoirs for all terms
        self.reservoirs: Dict[str, TermReservoir] = {
            t: TermReservoir(
                term=t,
                capacity=capacity_b,
                rng=random.Random(derive_term_seed(seed, t)),
            )
            for t in self.pool_terms
        }

        # Lazy components
        self._tokenizer = tokenizer
        self._model = model
        self._analyzer = analyzer

    @staticmethod
    def _load_canonical_pool(pool_path: str) -> List[str]:
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"Canonical pool file not found: {pool_path}")
        with open(pool_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            terms = data
        elif isinstance(data, dict) and "terms" in data:
            terms = data["terms"]
        elif isinstance(data, dict) and "pool_terms" in data:
            terms = data["pool_terms"]
        elif isinstance(data, dict):
            terms = list(data.keys())
        else:
            raise ValueError(f"Unrecognized pool format in {pool_path}")

        # Ensure deterministic ordering
        terms = sorted(list(dict.fromkeys(terms)))
        return terms

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
            )
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.model_name,
                revision=self.revision,
            )
            self._model.eval()
        return self._model

    @property
    def analyzer(self) -> Any:
        if self._analyzer is None:
            from evaluation.baselines.pyterrier_qe import get_terrier_analyzer

            self._analyzer = get_terrier_analyzer()
        return self._analyzer

    def finalize_sidecar(
        self,
        device: Optional[str] = None,
        force_cpu: bool = False,
    ) -> Dict[str, Any]:
        """Finalizes sidecar: deduplicates texts, batch encodes, and writes artifacts.

        Output artifacts:
        - chunks.parquet: Unique text table (embedding_idx, text, token_len)
        - chunk_embeddings.npy: Float16 memory-mapped array of shape (N_unique_texts, 384)
        - term_references.parquet: Exact source chunk metadata, pointers, and provenance
        - sidecar_manifest.json: Build provenance and metrics
        """
        os.makedirs(self.output_dir, exist_ok=True)
        t0 = time.time()

        if device is None:
            if not force_cpu and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if force_cpu and self.compute_dtype == "float16":
            raise ValueError(
                "Conflicting runtime override: cannot force CPU execution on sidecar "
                "initialized with compute_dtype='float16'. Dtype must be resolved upfront."
            )
        if device == "cpu" and self.compute_dtype == "float16":
            raise ValueError(
                "Conflicting runtime override: cannot run on CPU device with compute_dtype='float16'."
            )

        # 1. Deduplicate unique chunk texts across all surviving reservoirs
        unique_texts: Dict[str, int] = {}
        ordered_terms = sorted(self.pool_terms)

        for t in ordered_terms:
            res = self.reservoirs[t]
            for c in res.sample:
                if c.text not in unique_texts:
                    unique_texts[c.text] = len(unique_texts)

        num_unique_texts = len(unique_texts)
        ordered_texts = list(unique_texts.keys())

        # 2. Batch encode unique chunk texts on device
        if num_unique_texts > 0:
            embeddings_matrix, enc_time = self._encode_texts(
                ordered_texts,
                device=device,
                batch_size=self.batch_size,
            )
        else:
            embeddings_matrix = np.empty((0, BGE_EMBEDDING_DIM), dtype=np.float16)
            enc_time = 0.0

        t_ser_0 = time.time()
        # Save chunk_embeddings.npy
        emb_path = os.path.join(self.output_dir, "chunk_embeddings.npy")
        np.save(emb_path, embeddings_matrix)

        # Save chunks.parquet
        chunks_df = pd.DataFrame(
            {
                "embedding_idx": np.arange(num_unique_texts, dtype=np.int32),
                "text": ordered_texts,
                "token_len": [len(self.tokenizer(txt, add_special_tokens=False)["input_ids"]) for txt in ordered_texts],
            }
        )
        chunks_path = os.path.join(self.output_dir, "chunks.parquet")
        chunks_df.to_parquet(chunks_path, index=False)

        # 3. Assemble term_references table preserving source chunk identity and provenance
        records: List[Dict[str, Any]] = []
        for t in ordered_terms:
            res = self.reservoirs[t]
            chunk_ids = [c.chunk_id for c in res.sample]
            doc_ids = [c.doc_id for c in res.sample]
            chunk_indices = [unique_texts[c.text] for c in res.sample]
            char_starts = [c.char_start for c in res.sample]
            char_ends = [c.char_end for c in res.sample]
            tok_counts = [c.tok_count for c in res.sample]

            records.append(
                {
                    "term": t,
                    "population_N": res.population_N,
                    "sample_n": res.sample_n,
                    "distinct_docs_S": res.distinct_docs_S,
                    "status": res.status,
                    "chunk_ids": chunk_ids,
                    "doc_ids": doc_ids,
                    "chunk_indices": chunk_indices,
                    "char_starts": char_starts,
                    "char_ends": char_ends,
                    "tok_counts": tok_counts,
                }
            )

        refs_df = pd.DataFrame(records)
        refs_path = os.path.join(self.output_dir, "term_references.parquet")
        refs_df.to_parquet(refs_path, index=False)
        ser_time = time.time() - t_ser_0

        # Compute artifact SHA256 checksums
        artifact_checksums = {
            "chunk_embeddings.npy": compute_file_sha256(emb_path),
            "chunks.parquet": compute_file_sha256(chunks_path),
            "term_references.parquet": compute_file_sha256(refs_path),
        }

        # 4. Generate manifest with complete cost profiling and provenance
        total_finalize_time = time.time() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        )

        manifest = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "pool_path": self.pool_path,
            "pool_sha256": self.pool_sha256,
            "config_hash": self.config_hash,
            "analyzer_config": self.analyzer_config,
            "artifact_checksums": artifact_checksums,
            "corpus_completed": self.corpus_completed,
            "documents_processed": self.documents_processed,
            "max_docs": self.max_docs,
            "pool_terms_count": len(self.pool_terms),
            "verified_terms_count": int((refs_df["status"] == "VERIFIED").sum()),
            "unverified_terms_count": int((refs_df["status"] == "UNVERIFIED").sum()),
            "unique_chunks_encoded": num_unique_texts,
            "total_sampled_chunk_references": int(refs_df["sample_n"].sum()),
            "embedding_dim": BGE_EMBEDDING_DIM,
            "compute_dtype": self.compute_dtype,
            "storage_dtype": "float16",
            "model_name": self.model_name,
            "model_revision": self.revision,
            "seed": self.seed,
            "target_subwords": self.target_subwords,
            "max_subwords": self.max_subwords,
            "snap_window": self.snap_window,
            "capacity_b": self.capacity_b,
            "batch_size": self.batch_size,
            "ingestion_time_seconds": self.ingestion_time_seconds,
            "encoding_time_seconds": enc_time,
            "serialization_time_seconds": ser_time,
            "finalize_time_seconds": total_finalize_time,
            "total_build_seconds": self.ingestion_time_seconds + total_finalize_time,
            "peak_rss_mb": peak_rss_mb,
            "peak_vram_mb": peak_vram_mb,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        manifest_path = os.path.join(self.output_dir, "sidecar_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _encode_texts(
        self,
        texts: List[str],
        device: str,
        batch_size: int,
        compute_dtype: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Batch encodes text chunks into preallocated float16 array with L2 normalization."""
        t_enc_0 = time.time()
        dtype_str = compute_dtype or self.compute_dtype
        model = self.model

        orig_device = None
        orig_dtype = None
        try:
            param = next(model.parameters())
            orig_device = param.device
            orig_dtype = param.dtype
        except (StopIteration, AttributeError):
            pass

        if device == "cuda" and dtype_str == "float16":
            model = model.half().to(device)
        else:
            model = model.float().to(device)
        model.eval()

        n_texts = len(texts)
        embeddings_matrix = np.empty((n_texts, BGE_EMBEDDING_DIM), dtype=np.float16)

        try:
            with torch.no_grad():
                for i in range(0, n_texts, batch_size):
                    batch_texts = texts[i : i + batch_size]
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=False,  # Strict: fail loudly if chunk exceeds model limit
                        return_tensors="pt",
                    )

                    batch_tok_len = int(encoded["input_ids"].shape[1])
                    if batch_tok_len > MAX_ENCODER_TOKENS:
                        raise ValueError(
                            f"Batch token length {batch_tok_len} exceeds MAX_ENCODER_TOKENS ({MAX_ENCODER_TOKENS}). "
                            "Chunk segmentation violated encoder bounds!"
                        )
                    if batch_tok_len > self.max_subwords + 2:
                        raise ValueError(
                            f"Batch token length {batch_tok_len} exceeds max_subwords + 2 ({self.max_subwords + 2}). "
                            "Chunk segmentation violated chunk size bounds!"
                        )

                    inputs = {
                        k: v.to(device) if hasattr(v, "to") else v
                        for k, v in encoded.items()
                    }

                    outputs = model(**inputs)
                    # CLS pooling
                    cls_reps = outputs[0][:, 0]
                    # L2 normalize
                    norm_reps = torch.nn.functional.normalize(cls_reps, p=2, dim=1)

                    if device == "cuda" and dtype_str == "float16":
                        embeddings_matrix[i : i + len(batch_texts)] = norm_reps.half().cpu().numpy()
                    else:
                        embeddings_matrix[i : i + len(batch_texts)] = norm_reps.float().numpy().astype(np.float16)
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
            if orig_device is not None and orig_dtype is not None and hasattr(model, "to"):
                try:
                    model.to(device=orig_device, dtype=orig_dtype)
                except Exception:
                    pass

        enc_time = time.time() - t_enc_0
        return embeddings_matrix, enc_time


class ContextSidecar:
    """Zero-copy memory-mapped runtime container for Gate 2 context scoring.

    Provides O(1) term metadata lookups and zero-copy chunk embedding retrieval.
    """

    def __init__(
        self,
        sidecar_dir: str,
        expected_pool_sha256: Optional[str] = None,
        expected_corpus_sha256: Optional[str] = None,
        expected_config_hash: Optional[str] = None,
        expected_compute_dtype: Optional[str] = None,
        verify_checksums: bool = False,
        allow_incomplete: bool = False,
    ):
        self.sidecar_dir = sidecar_dir
        self.emb_path = os.path.join(sidecar_dir, "chunk_embeddings.npy")
        self.refs_path = os.path.join(sidecar_dir, "term_references.parquet")
        self.manifest_path = os.path.join(sidecar_dir, "sidecar_manifest.json")

        # Load manifest strictly
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Sidecar manifest missing at: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)

        if not allow_incomplete and not self.manifest.get("corpus_completed", False):
            raise ValueError(
                f"Sidecar at {sidecar_dir} is incomplete (corpus_completed is False). "
                "Incomplete caches cannot be used for experiments."
            )

        schema_ver = str(self.manifest.get("schema_version", "1.0.0"))
        if not schema_ver.startswith("2."):
            raise ValueError(
                f"Sidecar at {sidecar_dir} has incompatible schema version '{schema_ver}'. "
                f"Expected version 2.x (current: {SIDECAR_SCHEMA_VERSION}). Rebuild required."
            )

        if expected_pool_sha256 is not None:
            actual_pool = self.manifest.get("pool_sha256")
            if actual_pool != expected_pool_sha256:
                raise ValueError(
                    f"Pool hash mismatch in {sidecar_dir}: expected {expected_pool_sha256}, got {actual_pool}"
                )

        if expected_corpus_sha256 is not None:
            actual_corpus = self.manifest.get("corpus_sha256")
            if actual_corpus != expected_corpus_sha256:
                raise ValueError(
                    f"Corpus hash mismatch in {sidecar_dir}: expected {expected_corpus_sha256}, got {actual_corpus}"
                )

        if expected_config_hash is not None:
            actual_cfg = self.manifest.get("config_hash")
            if actual_cfg != expected_config_hash:
                raise ValueError(
                    f"Config hash mismatch in {sidecar_dir}: expected {expected_config_hash}, got {actual_cfg}"
                )

        if expected_compute_dtype is not None:
            actual_dtype = self.manifest.get("compute_dtype")
            if actual_dtype != expected_compute_dtype:
                raise ValueError(
                    f"Compute dtype mismatch in {sidecar_dir}: expected {expected_compute_dtype}, got {actual_dtype}"
                )

        if verify_checksums:
            recorded = self.manifest.get("artifact_checksums", {})
            for fname in ["chunk_embeddings.npy", "chunks.parquet", "term_references.parquet"]:
                fpath = os.path.join(sidecar_dir, fname)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Artifact {fname} missing in {sidecar_dir}")
                actual_sha = compute_file_sha256(fpath)
                exp_sha = recorded.get(fname)
                if exp_sha != actual_sha:
                    raise ValueError(
                        f"Checksum mismatch for {fname}: expected {exp_sha}, calculated {actual_sha}"
                    )

        if not os.path.exists(self.emb_path):
            raise FileNotFoundError(f"Embeddings file missing: {self.emb_path}")
        if not os.path.exists(self.refs_path):
            raise FileNotFoundError(f"Term references missing: {self.refs_path}")

        # Memory-map embeddings in read-only mode (zero-copy)
        self.embeddings: np.ndarray = np.load(self.emb_path, mmap_mode="r")

        # Load term references table into fast lookup
        df = pd.read_parquet(self.refs_path)
        records = df.to_dict(orient="records")
        self._term_lookup: Dict[str, Dict[str, Any]] = {
            r["term"]: {
                "population_N": int(r["population_N"]),
                "sample_n": int(r["sample_n"]),
                "distinct_docs_S": int(r["distinct_docs_S"]),
                "status": str(r["status"]),
                "chunk_ids": list(r["chunk_ids"]),
                "doc_ids": list(r["doc_ids"]),
                "chunk_indices": list(r["chunk_indices"]),
                "char_starts": list(r["char_starts"]),
                "char_ends": list(r["char_ends"]),
                "tok_counts": list(r["tok_counts"]),
            }
            for r in records
        }

    def __len__(self) -> int:
        return len(self._term_lookup)

    def __contains__(self, term: str) -> bool:
        return term in self._term_lookup

    def get_term_metadata(self, term: str) -> Optional[Dict[str, Any]]:
        """Returns metadata for a candidate term, or None if absent from pool."""
        return self._term_lookup.get(term)
