"""
Gate 1 sidecar manager: the offline assets of the two Gate 1 channels of CRVE.
1. Bounded PPMI sidecar: the top M = 600 co-occurring pool terms (positive PMI, exact floats) of every eligible anchor.
2. Sparse lexical profiles: per pool term up to R = 50 windows sampled by reservoir sampling over its occurrences, each
   with up to 32 analysed terms before and 31 after the occurrence (the centre occurrence of the term removed), and a
   BM25 index over these profiles.

- Provenance metadata (pool_sha256, num_docs, analyzer_version, corpus and lexicon hashes, created_at); a cache whose
  metadata does not match the pool, the index or the corpus is rejected.
- Build guards: time limits, a memory watchdog (abort at 12 GiB) and disk caps; files are written to .tmp and renamed.
"""

import os
import time
import math
import json
import shutil
import hashlib
import random
from typing import Dict, List, Tuple, Optional, Any, Iterator
from collections import defaultdict

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyterrier as pt

from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer

ANALYZER_VERSION = "v1_krovetz_suppletion"
DEFAULT_BUILD_CEILINGS_SEC = {
    "ppmi": 3600.0,     # 60 min
    "lexical": 2700.0,  # 45 min
}
DEFAULT_MAX_DISK_MB = {
    "ppmi": 250.0,
    "lexical": 350.0,
}
MAX_DISK_MB = 500.0


class BuildResourceError(RuntimeError):
    """Raised when sidecar build exceeds resource limits."""
    pass


def validate_gate1_config(config: Dict[str, Any]) -> None:
    """
    Fail-closed schema validation of the Gate 1 configuration (configs/gate1.yaml).
    Raises KeyError or ValueError if any required key is missing or invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("FATAL: Configuration must be a dictionary!")

    # 1. Memory Watchdog
    if "memory_watchdog" not in config:
        raise KeyError("FATAL: Missing required config section 'memory_watchdog'!")
    mw = config["memory_watchdog"]
    for key in ("hard_abort_rss_gib",):
        if key not in mw:
            raise KeyError(f"FATAL: Missing 'memory_watchdog.{key}'!")
        if not isinstance(mw[key], (int, float)) or mw[key] <= 0:
            raise ValueError(f"FATAL: 'memory_watchdog.{key}' must be a positive number!")

    # 2. Build Ceilings & Disk Caps
    for sec, cap_type in (("build_ceilings_sec", "seconds"), ("build_disk_caps_mb", "MB")):
        if sec not in config:
            raise KeyError(f"FATAL: Missing required config section '{sec}'!")
        for sidecar in ("ppmi_sidecar", "lexical_profiles"):
            if sidecar not in config[sec]:
                raise KeyError(f"FATAL: Missing '{sec}.{sidecar}'!")
            if not isinstance(config[sec][sidecar], (int, float)) or config[sec][sidecar] <= 0:
                raise ValueError(f"FATAL: '{sec}.{sidecar}' must be a positive number of {cap_type}!")

    # 3. Sidecars & Reservoir Seed
    if "sidecars" not in config:
        raise KeyError("FATAL: Missing required config section 'sidecars'!")
    sc = config["sidecars"]
    if "lexical_profiles" not in sc or "reservoir_seed" not in sc["lexical_profiles"]:
        raise KeyError("FATAL: Missing 'sidecars.lexical_profiles.reservoir_seed'!")
    if not isinstance(sc["lexical_profiles"]["reservoir_seed"], int):
        raise ValueError("FATAL: 'sidecars.lexical_profiles.reservoir_seed' must be an integer!")


def get_process_rss_gib() -> float:
    """Returns the total RSS of the current process and all its children in GiB."""
    try:
        parent = psutil.Process()
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 ** 3)
    except Exception:
        return 0.0


def check_build_watchdog(abort_rss_gib: float = 12.0):
    """Checks process tree RSS and aborts if it exceeds the hard cap."""
    rss = get_process_rss_gib()
    if rss >= abort_rss_gib:
        raise BuildResourceError(
            f"FATAL: Process RSS ({rss:.2f} GiB) exceeded hard cap of {abort_rss_gib} GiB during sidecar build!"
        )


def compute_pool_sha256(terms: List[str]) -> str:
    """Computes exact SHA-256 over newline-delimited terms without trailing newline."""
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


def compute_corpus_source_hash(dataset: str) -> str:
    """Computes SHA-256 hash over raw corpus files consumed by stream_corpus(). Raises on missing files."""
    paths = BenchmarkLoader.get_corpus_source_paths(dataset)
    if not paths:
        raise FileNotFoundError(f"FATAL: No corpus source paths found for dataset '{dataset}'!")
    h = hashlib.sha256()
    for p in sorted(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f"FATAL: Required corpus source file missing: {p}")
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    return h.hexdigest()


def compute_lexicon_semantic_hash(index) -> str:
    """Computes deterministic SHA-256 over sorted (term, df, cf) records from index lexicon."""
    lex = index.getLexicon()
    entries = []
    for entry in lex:
        t = str(entry.getKey())
        df = int(entry.getValue().getDocumentFrequency())
        cf = int(entry.getValue().getFrequency())
        entries.append(f"{t}\t{df}\t{cf}")
    entries.sort()
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


class Gate1SidecarManager:
    """Builds, caches and loads the Gate 1 sidecars with strict validation."""

    def __init__(
        self,
        cache_dir: str = "data/cache/canonical_pools",
        config: Optional[Dict[str, Any]] = None,
        disallow_rebuild: bool = False,
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.analyzer = get_terrier_analyzer()
        self.config = config or {}
        self.disallow_rebuild = disallow_rebuild

        if self.config:
            validate_gate1_config(self.config)
            self.hard_abort_rss_gib = float(self.config["memory_watchdog"]["hard_abort_rss_gib"])
        else:
            self.hard_abort_rss_gib = 12.0

        # Wire build ceilings and disk caps directly from config if provided
        key_map = {
            "ppmi_sidecar": "ppmi",
            "lexical_profiles": "lexical",
        }
        self.ceilings = dict(DEFAULT_BUILD_CEILINGS_SEC)
        if "build_ceilings_sec" in self.config:
            for k, v in self.config["build_ceilings_sec"].items():
                self.ceilings[k] = float(v)
                if k in key_map:
                    self.ceilings[key_map[k]] = float(v)

        self.disk_caps = dict(DEFAULT_MAX_DISK_MB)
        if "build_disk_caps_mb" in self.config:
            for k, v in self.config["build_disk_caps_mb"].items():
                self.disk_caps[k] = float(v)
                if k in key_map:
                    self.disk_caps[key_map[k]] = float(v)

    def _get_safe_ds(self, dataset: str) -> str:
        return dataset.lower().replace("-", "_")

    def _check_disk_footprint(self, path: str, max_mb: Optional[float] = None):
        """Asserts written file or directory is under max_mb. Deletes path if cap exceeded."""
        cap = max_mb if max_mb is not None else MAX_DISK_MB
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 ** 2)
        elif os.path.isdir(path):
            size_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path)
                for f in files
            ) / (1024 ** 2)
        else:
            return
        if size_mb > cap:
            # Clean up failed/oversized artifact immediately
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception:
                pass
            raise BuildResourceError(
                f"Disk footprint for {path} ({size_mb:.2f} MB) exceeded cap of {cap} MB!"
            )

    # -------------------------------------------------------------------------
    def _save_ppmi_parquet(
        self,
        out_path: str,
        anchor_ppmi: Dict[str, List[Tuple[str, float]]],
        metadata: Dict[str, Any],
    ) -> None:
        anchor_list = []
        cand_list = []
        score_list = []
        for a, cands in anchor_ppmi.items():
            for c, s in cands:
                anchor_list.append(a)
                cand_list.append(c)
                score_list.append(float(s))

        table = pa.Table.from_pydict({
            "anchor_term": anchor_list,
            "candidate_term": cand_list,
            "score": pa.array(score_list, type=pa.float64()),
        })
        custom_meta = {b"sidecar_metadata": json.dumps(metadata).encode("utf-8")}
        table = table.replace_schema_metadata(custom_meta)

        tmp_path = out_path + ".tmp"
        pq.write_table(table, tmp_path, compression="zstd")
        self._check_disk_footprint(tmp_path, max_mb=self.disk_caps.get("ppmi", 250.0))
        os.replace(tmp_path, out_path)

    # -------------------------------------------------------------------------
    # Bounded PPMI sidecar (co-occurrence channel): top M = 600 pool terms per anchor, exact floats
    # -------------------------------------------------------------------------
    def build_or_load_bounded_ppmi_sidecar(
        self,
        dataset: str,
        index,
        pool_terms: List[str],
        num_docs: int,
        top_m: int = 600,
        force_rebuild: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Builds or loads precomputed top-M=600 PPMI neighbors per anchor with exact floats (Parquet format).
        """
        if timeout_sec is None:
            timeout_sec = self.ceilings.get("ppmi", DEFAULT_BUILD_CEILINGS_SEC["ppmi"])
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_bounded_ppmi.parquet")
        expected_sha = compute_pool_sha256(pool_terms)

        corpus_source_hash = compute_corpus_source_hash(dataset)
        lexicon_semantic_hash = compute_lexicon_semantic_hash(index)

        if not force_rebuild and os.path.exists(out_path):
            try:
                table = pq.read_table(out_path)
                schema_meta = table.schema.metadata or {}
                meta = {}
                if b"sidecar_metadata" in schema_meta:
                    meta = json.loads(schema_meta[b"sidecar_metadata"].decode("utf-8"))
                if (
                    meta.get("pool_sha256") == expected_sha
                    and meta.get("num_docs") == num_docs
                    and meta.get("analyzer_version") == ANALYZER_VERSION
                    and meta.get("top_m") == top_m
                    and meta.get("corpus_source_hash") == corpus_source_hash
                    and meta.get("lexicon_semantic_hash") == lexicon_semantic_hash
                ):
                    df_p = table.to_pandas()
                    anchor_ppmi = defaultdict(list)
                    for a, c, s in zip(df_p["anchor_term"], df_p["candidate_term"], df_p["score"]):
                        anchor_ppmi[a].append((c, float(s)))
                    return {
                        "anchor_ppmi": dict(anchor_ppmi),
                        "metadata": meta,
                        "timing_s": 0.0,
                    }
                else:
                    if self.disallow_rebuild:
                        raise RuntimeError(f"FATAL: PPMI sidecar cache invalid or outdated for {dataset}. Auto-rebuild is disallowed (disallow_rebuild=True).")
                    print(f"[Sidecar] PPMI cache invalid or outdated for {dataset}. Rebuilding...")
            except Exception as e:
                if self.disallow_rebuild:
                    raise RuntimeError(f"FATAL: Failed to load PPMI cache for {dataset} ({e}). Auto-rebuild is disallowed (disallow_rebuild=True).")
                print(f"[Sidecar] Failed to load PPMI cache for {dataset} ({e}). Rebuilding...")
        elif not force_rebuild and self.disallow_rebuild:
            raise RuntimeError(f"FATAL: Parquet PPMI sidecar missing at {out_path}. Auto-rebuild is disallowed (disallow_rebuild=True).")

        print(f"[Sidecar] Building Bounded PPMI Sidecar for {dataset} (top_m={top_m}, exact float, Parquet)...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)
        lex = index.getLexicon()

        df_map = {}
        for entry in lex:
            t = str(entry.getKey())
            df = int(entry.getValue().getDocumentFrequency())
            df_map[t] = df

        eligible_anchor_set = {
            t for t, df in df_map.items()
            if df >= 2 and (df / num_docs) <= 0.12
        }

        # Streaming co-occurrence accumulation with bounded memory
        anchor_cooccur: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            if doc_count % 2000 == 0:
                check_build_watchdog(self.hard_abort_rss_gib)
                if time.perf_counter() - t0 > timeout_sec:
                    raise TimeoutError(f"PPMI sidecar build exceeded time limit of {timeout_sec}s")

            terms, _ = self.analyzer.analyze(text)
            doc_terms = set(terms)
            doc_pool_terms = doc_terms.intersection(pool_set)
            if not doc_pool_terms:
                continue

            for a in doc_terms:
                if a in eligible_anchor_set:
                    a_counts = anchor_cooccur[a]
                    for t in doc_pool_terms:
                        if t != a:
                            a_counts[t] += 1

        # Track empirical peak memory and pair statistics
        peak_pairs = sum(len(v) for v in anchor_cooccur.values())
        rss_after_accum = get_process_rss_gib()
        running_peak_rss = max(get_process_rss_gib(), rss_after_accum)

        # Compute exact PPMI (no rounding)
        anchor_ppmi = {}
        for a, targets in anchor_cooccur.items():
            df_a = df_map.get(a, 0)
            if df_a < 2:
                continue
            scored = []
            for t, df_at in targets.items():
                if df_at >= 2:
                    df_t = df_map.get(t, 0)
                    if df_t >= 1:
                        pmi = math.log((num_docs * df_at) / (df_a * df_t))
                        if pmi > 0.0:
                            scored.append((t, float(pmi)))
            if scored:
                scored.sort(key=lambda x: (-x[1], x[0]))
                anchor_ppmi[a] = scored[:top_m]

        running_peak_rss = max(running_peak_rss, get_process_rss_gib())
        elapsed_sec = round(time.perf_counter() - t0, 2)
        metadata = {
            "dataset": dataset,
            "pool_sha256": expected_sha,
            "pool_size": len(pool_terms),
            "num_docs": num_docs,
            "analyzer_version": ANALYZER_VERSION,
            "top_m": top_m,
            "num_anchors": len(anchor_ppmi),
            "peak_pairs": peak_pairs,
            "peak_rss_gib": round(running_peak_rss, 3),
            "running_peak_rss_gib": round(running_peak_rss, 3),
            "rss_after_accumulation_gib": round(rss_after_accum, 3),
            "corpus_source_hash": corpus_source_hash,
            "lexicon_semantic_hash": lexicon_semantic_hash,
            "elapsed_sec": elapsed_sec,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Atomic write to Parquet
        self._save_ppmi_parquet(out_path, anchor_ppmi, metadata)

        timing_s = elapsed_sec
        print(f"[Sidecar] Bounded PPMI Sidecar built in {timing_s}s for {len(anchor_ppmi):,} anchors (peak pairs: {peak_pairs:,}, peak RSS: {running_peak_rss:.2f} GiB) -> {out_path}")

        return {
            "anchor_ppmi": anchor_ppmi,
            "metadata": metadata,
            "timing_s": timing_s,
        }

    def _save_lexical_parquet(
        self,
        out_path: str,
        docnos: List[str],
        texts: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        table = pa.Table.from_pydict({
            "docno": docnos,
            "text": texts,
        })
        custom_meta = {b"sidecar_metadata": json.dumps(metadata).encode("utf-8")}
        table = table.replace_schema_metadata(custom_meta)

        tmp_path = out_path + ".tmp"
        pq.write_table(table, tmp_path, compression="zstd")
        self._check_disk_footprint(tmp_path, max_mb=self.disk_caps.get("lexical", 350.0))
        os.replace(tmp_path, out_path)

    # -------------------------------------------------------------------------
    # Sparse lexical profiles (lexical profile channel): profiles in Parquet and their BM25 index
    # -------------------------------------------------------------------------
    def build_or_load_sparse_lexical_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        num_docs: int,
        force_rebuild: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Builds or loads auxiliary BM25 index over candidate terms' passage context profiles
        using deterministic reservoir sampling (up to 50 passages per candidate), persisted in Parquet.
        """
        if timeout_sec is None:
            timeout_sec = self.ceilings.get("lexical", DEFAULT_BUILD_CEILINGS_SEC["lexical"])
        safe_ds = self._get_safe_ds(dataset)
        parquet_path = os.path.join(self.cache_dir, f"{safe_ds}_lexical_profiles.parquet")
        aux_index_dir = os.path.join(self.cache_dir, f"{safe_ds}_lexical_profiles_idx")
        prop_path = os.path.join(aux_index_dir, "data.properties")
        meta_path = os.path.join(aux_index_dir, "sidecar_metadata.json")
        expected_sha = compute_pool_sha256(pool_terms)

        corpus_source_hash = compute_corpus_source_hash(dataset)
        configured_seed = (
            self.config.get("sidecars", {}).get("lexical_profiles", {}).get("reservoir_seed", 42)
            if self.config else 42
        )

        if not force_rebuild:
            if os.path.exists(parquet_path):
                try:
                    table = pq.read_table(parquet_path)
                    schema_meta = table.schema.metadata or {}
                    meta = {}
                    if b"sidecar_metadata" in schema_meta:
                        meta = json.loads(schema_meta[b"sidecar_metadata"].decode("utf-8"))
                    if (
                        meta.get("pool_sha256") == expected_sha
                        and meta.get("num_docs") == num_docs
                        and meta.get("analyzer_version") == ANALYZER_VERSION
                        and meta.get("corpus_source_hash") == corpus_source_hash
                        and meta.get("reservoir_seed") == configured_seed
                    ):
                        if not os.path.exists(prop_path):
                            if self.disallow_rebuild:
                                raise RuntimeError(
                                    f"FATAL: Lexical profiles auxiliary index missing for {dataset} at {aux_index_dir}. "
                                    f"Auto-rebuild is disallowed (disallow_rebuild=True)."
                                )
                            print(f"[Sidecar] Building auxiliary index from lexical profiles Parquet for {dataset}...")
                            df_lex = table.to_pandas()
                            def doc_gen() -> Iterator[Dict[str, str]]:
                                for d, txt in zip(df_lex["docno"], df_lex["text"]):
                                    yield {"docno": str(d), "text": str(txt)}
                            tmp_idx_dir = aux_index_dir + ".tmp"
                            if os.path.exists(tmp_idx_dir):
                                shutil.rmtree(tmp_idx_dir)
                            os.makedirs(tmp_idx_dir, exist_ok=True)
                            indexer = pt.IterDictIndexer(os.path.abspath(tmp_idx_dir), overwrite=True)
                            indexer.index(doc_gen())
                            with open(os.path.join(tmp_idx_dir, "sidecar_metadata.json"), "w", encoding="utf-8") as f:
                                json.dump(meta, f)
                            if os.path.exists(aux_index_dir):
                                shutil.rmtree(aux_index_dir)
                            os.rename(tmp_idx_dir, aux_index_dir)

                        retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
                        return {
                            "index_path": aux_index_dir,
                            "parquet_path": parquet_path,
                            "retriever": retriever,
                            "metadata": meta,
                            "timing_s": 0.0,
                        }
                    else:
                        if self.disallow_rebuild:
                            raise RuntimeError(
                                f"FATAL: Lexical profiles cache invalid or outdated for {dataset}. "
                                f"Auto-rebuild is disallowed (disallow_rebuild=True)."
                            )
                        print(f"[Sidecar] Lexical profiles cache invalid or outdated for {dataset}. Rebuilding...")
                except Exception as e:
                    if self.disallow_rebuild:
                        raise RuntimeError(
                            f"FATAL: Failed to load Lexical profiles cache for {dataset} ({e}). "
                            f"Auto-rebuild is disallowed (disallow_rebuild=True)."
                        )
                    print(f"[Sidecar] Failed to load Lexical profiles cache for {dataset} ({e}). Rebuilding...")
            elif os.path.exists(prop_path) and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if (
                        meta.get("pool_sha256") == expected_sha
                        and meta.get("num_docs") == num_docs
                        and meta.get("analyzer_version") == ANALYZER_VERSION
                        and meta.get("corpus_source_hash") == corpus_source_hash
                        and meta.get("reservoir_seed") == configured_seed
                    ):
                        retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
                        return {
                            "index_path": aux_index_dir,
                            "parquet_path": None,
                            "retriever": retriever,
                            "metadata": meta,
                            "timing_s": 0.0,
                        }
                    else:
                        if self.disallow_rebuild:
                            raise RuntimeError(
                                f"FATAL: Lexical profiles cache invalid or outdated for {dataset}. "
                                f"Auto-rebuild is disallowed (disallow_rebuild=True)."
                            )
                        print(f"[Sidecar] Lexical profiles cache invalid or outdated for {dataset}. Rebuilding...")
                except Exception as e:
                    if self.disallow_rebuild:
                        raise RuntimeError(
                            f"FATAL: Failed to load Lexical profiles cache for {dataset} ({e}). "
                            f"Auto-rebuild is disallowed (disallow_rebuild=True)."
                        )
                    print(f"[Sidecar] Failed to load Lexical profiles cache for {dataset} ({e}). Rebuilding...")
            elif self.disallow_rebuild:
                raise RuntimeError(
                    f"FATAL: Lexical profiles sidecar missing at {parquet_path} / {aux_index_dir}. "
                    f"Auto-rebuild is disallowed (disallow_rebuild=True)."
                )

        print(f"[Sidecar] Building Sparse Lexical Context Sidecar for {dataset} ({len(pool_terms)} terms, reservoir sampling, seed={configured_seed})...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        # Reservoir sampling: reservoir size K=50 per pool term with configured seed
        rng = random.Random(configured_seed)
        reservoirs: Dict[str, List[str]] = defaultdict(list)
        item_counts: Dict[str, int] = defaultdict(int)

        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            if doc_count % 2000 == 0:
                check_build_watchdog(self.hard_abort_rss_gib)
                if time.perf_counter() - t0 > timeout_sec:
                    raise TimeoutError(f"Lexical sidecar build exceeded time limit of {timeout_sec}s")

            terms, _ = self.analyzer.analyze(text)
            doc_terms = set(terms)
            doc_pool_terms = doc_terms.intersection(pool_set)
            if not doc_pool_terms:
                continue

            tok_len = len(terms)
            for t in doc_pool_terms:
                for idx, tok in enumerate(terms):
                    if tok == t:
                        item_counts[t] += 1
                        n = item_counts[t]
                        start = max(0, idx - 32)
                        end = min(tok_len, idx + 32)
                        passage_str = " ".join([w for j, w in enumerate(terms[start:end]) if j != (idx - start)])

                        if len(reservoirs[t]) < 50:
                            reservoirs[t].append(passage_str)
                        else:
                            # Standard Algorithm R reservoir replacement
                            j = rng.randint(0, n - 1)
                            if j < 50:
                                reservoirs[t][j] = passage_str

        # Write metadata
        metadata = {
            "dataset": dataset,
            "pool_sha256": expected_sha,
            "corpus_source_hash": corpus_source_hash,
            "pool_size": len(pool_terms),
            "num_docs": num_docs,
            "analyzer_version": ANALYZER_VERSION,
            "passages_per_candidate": 50,
            "reservoir_seed": configured_seed,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Collect docnos and texts
        docnos = []
        texts = []
        for t in pool_terms:
            passages = reservoirs.get(t, [])
            combined_text = " ".join(passages) if passages else t
            docnos.append(t)
            texts.append(combined_text)

        # Write Parquet table
        self._save_lexical_parquet(parquet_path, docnos, texts, metadata)

        # Stream documents to IterDictIndexer
        def doc_generator() -> Iterator[Dict[str, str]]:
            for d, txt in zip(docnos, texts):
                yield {"docno": d, "text": txt}

        tmp_idx_dir = aux_index_dir + ".tmp"
        if os.path.exists(tmp_idx_dir):
            shutil.rmtree(tmp_idx_dir)
        os.makedirs(tmp_idx_dir, exist_ok=True)

        indexer = pt.IterDictIndexer(os.path.abspath(tmp_idx_dir), overwrite=True)
        indexer.index(doc_generator())

        with open(os.path.join(tmp_idx_dir, "sidecar_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        self._check_disk_footprint(tmp_idx_dir, max_mb=self.disk_caps.get("lexical", 350.0))

        if os.path.exists(aux_index_dir):
            shutil.rmtree(aux_index_dir)
        os.rename(tmp_idx_dir, aux_index_dir)

        retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Sparse Lexical Sidecar built in {timing_s}s -> {parquet_path} & {aux_index_dir}")

        return {
            "index_path": aux_index_dir,
            "parquet_path": parquet_path,
            "retriever": retriever,
            "metadata": metadata,
            "timing_s": timing_s,
        }
