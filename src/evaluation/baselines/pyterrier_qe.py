"""
Query-expansion baselines for PyTerrier:
1. BGE_Vocab_QE (BGE-QE): embedding-based expansion over a static vocabulary (the 15,000 most frequent terms of the
   collection, BAAI/bge-small-en-v1.5 vectors of their most frequent surface forms): 50 nearest neighbours per query term,
   union scored by the qtf-weighted mean cosine, the ten best added with a total weight of (1 - 0.6) / 0.6 times the query.
2. LLM_Q2E_ZS (LLM-Q2E): zero-shot keyword expansion with a local LLM (qwen3.5:4b through Ollama), keywords appended to
   five copies of the analysed query; generations cached under data/cache/qe_llm/raw.

Both rewriters map their terms to Terrier lexicon terms (Porter stems); a query without valid expansion terms is left
unchanged.
"""

import os
import time
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import requests
import re

import pyterrier as pt

# The settings of BGE-QE and LLM-Q2E used in the paper
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # BAAI/bge-small-en-v1.5, the revision of CRVE's encoder
DEFAULT_QE_CONFIG = {
    "bge_vocab_qe": {
        "encoder": "BAAI/bge-small-en-v1.5",
        "vocab_cap": 15000,
        "min_cf": 3,
        "neighbour_depth_L": 50,
        "expansion_terms_ke": 10,
        "alpha": 0.60,
        "cache_dir": "data/cache/qe_vocab",
        "batch_size": 64,
        "seed": 42,
    },
    "llm_q2e_zs": {
        "model_profile": "qwen3.5-4b",
        "backend": "ollama",
        "endpoint": "http://localhost:11434/api/generate",
        "tag": "qwen3.5:4b",
        "prompt_template": "Write only a comma-separated list of keywords for the following query. Do not output any intro, markdown, categories, or explanations:\n{query}",
        "temperature": 0.0,
        "max_tokens": 64,
        "repetition_r": 5,
        "seed": 42,
        "think": False,
        "raw_cache_dir": "data/cache/qe_llm/raw",
        "validated_cache_dir": "data/cache/qe_llm/validated",
    },
}


class TerrierQueryAnalyzer:
    """
    Analyzes queries and raw text into Terrier lexicon terms (Porter stems)
    and extracts natural token surfaces for neural embedding.
    """
    def __init__(self):
        if not pt.java.started():
            pt.java.init()
        self.StringReader = pt.java.autoclass("java.io.StringReader")
        self.Tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
        self.Stopwords = pt.java.autoclass("org.terrier.terms.Stopwords")(None)
        self.PorterStemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()

    def analyze(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Returns (analyzed_terms, natural_surfaces) where:
        - analyzed_terms are non-stopword Porter-stemmed terms recognized by Terrier.
        - natural_surfaces are the corresponding natural English token strings.
        """
        if not text or not text.strip():
            return [], []
        stream = self.Tokeniser.tokenise(self.StringReader(text))
        terms = []
        surfaces = []
        while stream.hasNext():
            s = str(stream.next())
            if not s:
                continue
            s_lower = s.lower()
            if self.Stopwords.isStopword(s_lower):
                continue
            term = self.PorterStemmer.stem(s_lower)
            if term:
                terms.append(str(term))
                surfaces.append(str(s))
        return terms, surfaces

    def to_query_toks(self, terms: List[str]) -> Dict[str, float]:
        """Converts a list of analyzed terms into a term-frequency dictionary."""
        counts = Counter(terms)
        return {k: float(v) for k, v in counts.items()}


_ANALYZER_INSTANCE: Optional[TerrierQueryAnalyzer] = None

def get_terrier_analyzer() -> TerrierQueryAnalyzer:
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None:
        _ANALYZER_INSTANCE = TerrierQueryAnalyzer()
    return _ANALYZER_INSTANCE


class BGEVocabSidecarManager:
    """
    Manages static vocabulary extraction and BGE embedding sidecars:
    data/cache/qe_vocab/{dataset}/{cache_identity}/
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_QE_CONFIG["bge_vocab_qe"]
        self.cache_dir = self.config.get("cache_dir", "data/cache/qe_vocab")
        self.vocab_cap = self.config.get("vocab_cap", 15000)
        self.min_cf = self.config.get("min_cf", 3)
        self.encoder_name = self.config.get("encoder", "BAAI/bge-small-en-v1.5")
        self._encoder_model = None

    def _get_encoder(self):
        if self._encoder_model is None:
            from sentence_transformers import SentenceTransformer
            self._encoder_model = SentenceTransformer(self.encoder_name, revision=BGE_REVISION, device="cuda")
        return self._encoder_model

    def compute_cache_identity(self, dataset_name: str, index_fingerprint: str) -> str:
        s = f"{dataset_name}:{index_fingerprint}:{self.encoder_name}:{self.vocab_cap}:{self.min_cf}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def build_or_load_sidecar(self, dataset_name: str, index) -> Dict[str, Any]:
        """
        Builds or loads the 15,000-term vocabulary sidecar:
        - display_surfaces (natural words)
        - retrieval_terms (Terrier Porter stems)
        - embeddings (FP16 normalized BGE vectors: [15000, 384])
        - metadata & manifest
        """
        safe_ds = dataset_name.lower().replace("-", "_")
        index_fp = str(index.getCollectionStatistics().getNumberOfDocuments())
        cache_id = self.compute_cache_identity(safe_ds, index_fp)
        sidecar_dir = os.path.join(self.cache_dir, safe_ds, cache_id)
        manifest_path = os.path.join(sidecar_dir, "manifest.json")
        vecs_path = os.path.join(sidecar_dir, "embeddings.npy")
        vocab_path = os.path.join(sidecar_dir, "vocab.parquet")

        timing = {
            "qe_lexicon_extract_s": 0.0,
            "qe_surface_recovery_s": 0.0,
            "qe_prepare_s": 0.0,
            "qe_cache_hit": False,
        }

        if os.path.exists(manifest_path) and os.path.exists(vecs_path) and os.path.exists(vocab_path):
            t0 = time.perf_counter()
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            df_vocab = pd.read_parquet(vocab_path)
            embeddings = np.load(vecs_path)
            timing["qe_prepare_s"] = round(time.perf_counter() - t0, 2)
            timing["qe_cache_hit"] = True
            return {
                "display_surfaces": df_vocab["display_surface"].tolist(),
                "retrieval_terms": df_vocab["retrieval_term"].tolist(),
                "df": df_vocab["df"].values,
                "cf": df_vocab["cf"].values,
                "embeddings": embeddings,
                "manifest": manifest,
                "timing": timing,
                "sidecar_mb": round(os.path.getsize(vecs_path) / (1024 ** 2), 2),
            }

        # Build Sidecar
        t_start = time.perf_counter()
        os.makedirs(sidecar_dir, exist_ok=True)
        print(f"[BGE-QE] Building static vocabulary sidecar for '{safe_ds}' -> {sidecar_dir}")

        # Step 1: Extract top 20,000 CF candidate terms from loaded Terrier lexicon
        t_lex = time.perf_counter()
        lex = index.getLexicon()
        candidates = []
        for entry in lex:
            term = str(entry.getKey())
            cf = int(entry.getValue().getFrequency())
            df = int(entry.getValue().getDocumentFrequency())
            if cf >= self.min_cf and len(term) >= 2 and not term.isdigit():
                candidates.append((term, cf, df))
        timing["qe_lexicon_extract_s"] = round(time.perf_counter() - t_lex, 2)

        # Sort descending by CF, tie-break lexically
        candidates.sort(key=lambda x: (-x[1], x[0]))
        candidates = candidates[:20000]
        candidate_terms_set = {c[0]: (c[1], c[2]) for c in candidates}
        print(f"  [Lexicon] Extracted {len(candidates)} candidate terms (CF >= {self.min_cf}) in {timing['qe_lexicon_extract_s']}s")

        # Step 2: Surface Recovery streaming pass (extract most frequent natural surface for each stem)
        from evaluation.benchmark_loader import BenchmarkLoader
        analyzer = get_terrier_analyzer()
        surface_counts: Dict[str, Counter] = {t: Counter() for t in candidate_terms_set}

        t_surf = time.perf_counter()
        doc_count = 0
        max_surface_sample = 5000  # surface forms are taken from the first 5,000 documents
        for doc_id, text in BenchmarkLoader.stream_corpus(dataset_name):
            doc_count += 1
            if not text:
                continue
            terms, surfs = analyzer.analyze(text)
            for t, s in zip(terms, surfs):
                if t in surface_counts:
                    surface_counts[t][s.lower()] += 1
            if doc_count >= max_surface_sample:
                break

        timing["qe_surface_recovery_s"] = round(time.perf_counter() - t_surf, 2)
        print(f"  [Surface Recovery] Sampled {doc_count:,} docs in {timing['qe_surface_recovery_s']}s")

        # Step 3: Retain top vocab_cap (15,000) valid terms with recovered surfaces
        valid_vocab = []
        for term, (cf, df) in candidate_terms_set.items():
            top_surfaces = surface_counts[term].most_common(1)
            display_surf = top_surfaces[0][0] if top_surfaces else term
            valid_vocab.append({
                "retrieval_term": term,
                "display_surface": display_surf,
                "cf": cf,
                "df": df,
            })
            if len(valid_vocab) >= self.vocab_cap:
                break

        df_vocab = pd.DataFrame(valid_vocab)
        df_vocab.to_parquet(vocab_path, index=False)

        # Step 4: Batch embed natural surfaces with frozen BGE-small
        encoder = self._get_encoder()
        surfaces_to_embed = df_vocab["display_surface"].tolist()
        print(f"  [BGE Encoder] Encoding {len(surfaces_to_embed)} vocabulary terms with {self.encoder_name}...")
        embeddings = encoder.encode(
            surfaces_to_embed,
            batch_size=self.config.get("batch_size", 64),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float16)

        np.save(vecs_path, embeddings)

        manifest = {
            "dataset": safe_ds,
            "index_fingerprint": index_fp,
            "encoder": self.encoder_name,
            "vocab_cap": self.vocab_cap,
            "min_cf": self.min_cf,
            "realized_vocab_count": len(valid_vocab),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        timing["qe_prepare_s"] = round(time.perf_counter() - t_start, 2)
        print(f"  [BGE-QE] Sidecar ready in {timing['qe_prepare_s']}s ({len(valid_vocab)} terms, {vecs_path})")

        return {
            "display_surfaces": df_vocab["display_surface"].tolist(),
            "retrieval_terms": df_vocab["retrieval_term"].tolist(),
            "df": df_vocab["df"].values,
            "cf": df_vocab["cf"].values,
            "embeddings": embeddings,
            "manifest": manifest,
            "timing": timing,
            "sidecar_mb": round(os.path.getsize(vecs_path) / (1024 ** 2), 2),
        }


class BGEVocabQERewriter(pt.Transformer):
    """
    PyTerrier Transformer for BGE_Vocab_QE:
    - Analyzes query text into aligned natural surfaces.
    - Reuses precomputed vocabulary embeddings for known surfaces (0ms neural overhead).
    - Encodes only unseen surfaces with frozen BGE encoder.
    - Evaluates 1-pass GPU FP16 GEMM matrix multiplication: [M, V].
    - Retrieves top-L neighbours via GPU torch.topk.
    - Selects candidate union C(Q), excluding original query terms.
    - Computes qtf-weighted mean cosine similarity on GPU.
    - Allocates B_alpha(Q) expansion mass across top ke candidates.
    - Emits query_toks for downstream Terrier BM25 matching.
    """
    def __init__(self, sidecar: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.sidecar = sidecar
        self.config = config or DEFAULT_QE_CONFIG["bge_vocab_qe"]
        self.L = self.config.get("neighbour_depth_L", 50)
        self.ke = self.config.get("expansion_terms_ke", 10)
        self.alpha = self.config.get("alpha", 0.60)
        self.encoder_name = self.config.get("encoder", "BAAI/bge-small-en-v1.5")
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = sidecar["embeddings"]
        # Convert embeddings to PyTorch Tensor on GPU in FP16 (takes only ~11.5 MB VRAM)
        if self.device == "cuda":
            self.gpu_embeddings = torch.tensor(self.embeddings, device="cuda", dtype=torch.float16)
            self.gpu_embeddings = torch.nn.functional.normalize(self.gpu_embeddings, p=2, dim=-1)
        else:
            self.gpu_embeddings = torch.tensor(self.embeddings, device="cpu", dtype=torch.float32)
            self.gpu_embeddings = torch.nn.functional.normalize(self.gpu_embeddings, p=2, dim=-1)

        self.retrieval_terms = list(sidecar["retrieval_terms"])
        self.display_surfaces = list(sidecar.get("display_surfaces", []))
        self.term_to_idx = {t: i for i, t in enumerate(self.retrieval_terms)}
        # Map known natural surfaces to index in vocabulary matrix for 0ms lookup
        self.surf_to_idx = {str(s).lower(): i for i, s in enumerate(self.display_surfaces)}
        # Persistent GPU embedding cache for query surfaces not in the static 15,000-term vocabulary
        self.surface_cache = {}

        self.analyzer = get_terrier_analyzer()
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name, revision=BGE_REVISION, device=self.device)
        return self._encoder

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input queries DataFrame by computing BGE_Vocab_QE query_toks."""
        res_df = df.copy()
        import torch

        new_query_toks = []
        new_queries = []

        # Batch distinct query surfaces across chunk
        chunk_queries = res_df["query"].tolist()
        analyzed_queries = []
        needed_surfs = {}

        for q_text in chunk_queries:
            terms, surfs = self.analyzer.analyze(q_text)
            analyzed_queries.append((terms, surfs))
            for s in surfs:
                s_lower = s.lower()
                # If surface is already in vocabulary sidecar or persistent cache, 0ms reuse!
                if (s_lower not in self.surf_to_idx and 
                    s_lower not in self.surface_cache and 
                    s_lower not in needed_surfs):
                    needed_surfs[s_lower] = s

        # Only encode unseen query surfaces with the neural network once
        unseen_keys = list(needed_surfs.keys())
        if unseen_keys:
            encoder = self._get_encoder()
            unseen_vectors = encoder.encode(
                unseen_keys,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for i, k in enumerate(unseen_keys):
                self.surface_cache[k] = torch.tensor(
                    unseen_vectors[i], device=self.device, dtype=self.gpu_embeddings.dtype
                )

        # Process each query using fast GPU tensor operations
        for idx, (terms, surfs) in enumerate(analyzed_queries):
            orig_toks = self.analyzer.to_query_toks(terms)
            if not orig_toks or not surfs:
                new_query_toks.append(orig_toks)
                new_queries.append(chunk_queries[idx])
                continue

            # Distinct natural surface representatives Au(Q)
            term_to_surfs = {}
            for t, s in zip(terms, surfs):
                term_to_surfs.setdefault(t, []).append(s.lower())

            au_surfs = []
            au_qtfs = []
            q_vec_list = []
            for t, surf_list in term_to_surfs.items():
                top_s = Counter(surf_list).most_common(1)[0][0]
                au_surfs.append(top_s)
                au_qtfs.append(orig_toks[t])
                # Check precomputed GPU embedding first (0ms lookup)
                if top_s in self.surf_to_idx:
                    q_vec_list.append(self.gpu_embeddings[self.surf_to_idx[top_s]])
                elif top_s in self.surface_cache:
                    q_vec_list.append(self.surface_cache[top_s])

            if len(q_vec_list) == 0:
                new_query_toks.append(orig_toks)
                new_queries.append(chunk_queries[idx])
                continue

            # Stack query vectors on GPU: [M, 384]
            q_tensor = torch.stack(q_vec_list)
            q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=-1)

            # High-speed GPU Matrix Multiplication: [M, V] (< 0.05 ms)
            sims = torch.matmul(q_tensor, self.gpu_embeddings.T)

            # Top-L neighbours per term on GPU
            top_k_val = min(self.L, sims.shape[1])
            top_scores, top_indices = torch.topk(sims, k=top_k_val, dim=-1)

            # Candidate union C(Q) of top-L neighbours, excluding original query terms
            orig_terms_set = set(orig_toks.keys())
            flat_indices = top_indices.view(-1).tolist()
            candidate_indices = []
            seen_cand = set()
            for c_idx in flat_indices:
                if c_idx not in seen_cand:
                    seen_cand.add(c_idx)
                    c_term = self.retrieval_terms[c_idx]
                    if c_term not in orig_terms_set:
                        candidate_indices.append(c_idx)

            if not candidate_indices:
                new_query_toks.append(orig_toks)
                new_queries.append(chunk_queries[idx])
                continue

            # 1-pass qtf-weighted mean similarity s(t, Q) across all vocabulary on GPU (< 0.01 ms)
            qtf_tensor = torch.tensor(au_qtfs, device=self.device, dtype=self.gpu_embeddings.dtype)  # [M]
            total_qtf = qtf_tensor.sum()
            weighted_sims = torch.matmul(qtf_tensor, sims) / total_qtf  # [V]

            cand_idx_tensor = torch.tensor(candidate_indices, device=self.device, dtype=torch.long)
            cand_mean_sims = weighted_sims[cand_idx_tensor].cpu().numpy()

            scored_candidates = []
            for i, c_idx in enumerate(candidate_indices):
                u_score = float(max(cand_mean_sims[i], 0.0))
                if u_score > 0:
                    scored_candidates.append((u_score, self.retrieval_terms[c_idx]))

            # Rank descending by score, tie-break lexically
            scored_candidates.sort(key=lambda x: (-x[0], x[1]))
            top_expanded = scored_candidates[:self.ke]

            total_u = sum(x[0] for x in top_expanded)
            if not top_expanded or total_u <= 0:
                new_query_toks.append(orig_toks)
                new_queries.append(chunk_queries[idx])
                continue

            # Mass allocation: B_alpha(Q) = ((1 - alpha) / alpha) * A(Q)
            a_q = sum(orig_toks.values())
            b_alpha = ((1.0 - self.alpha) / self.alpha) * a_q

            final_toks = dict(orig_toks)
            for u_score, c_term in top_expanded:
                delta_w = b_alpha * (u_score / total_u)
                final_toks[c_term] = float(final_toks.get(c_term, 0.0) + delta_w)

            new_query_toks.append(final_toks)
            new_queries.append(chunk_queries[idx])

        res_df["query_toks"] = new_query_toks
        return res_df


class LLMQEKeywordRewriter(pt.Transformer):
    """
    PyTerrier Transformer for LLM_Q2E_ZS:
    - Prompts local LLM (qwen3.5:4b) with exact Q2E/ZS zero-shot prompt.
    - Enforces thinking = False to eliminate latency.
    - Caches raw LLM outputs to disk.
    - Concatenates 5 copies of original query + generated keywords: Q' = Concat(5*Q, G(Q)).
    - Passes Q' through Terrier query analysis to construct query_toks.
    - Exact BM25 fallback if no valid lexicon term is generated.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or DEFAULT_QE_CONFIG["llm_q2e_zs"]
        self.endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
        self.model_tag = self.config.get("tag", "qwen3.5:4b")
        self.prompt_tmpl = self.config.get(
            "prompt_template",
            "Write only a comma-separated list of keywords for the following query. Do not output any intro, markdown, categories, or explanations:\n{query}"
        )
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.config.get("max_tokens", 64)
        self.repetition_r = self.config.get("repetition_r", 5)
        self.think = self.config.get("think", False)
        self.raw_cache_dir = self.config.get("raw_cache_dir", "data/cache/qe_llm/raw")
        self.analyzer = get_terrier_analyzer()
        os.makedirs(self.raw_cache_dir, exist_ok=True)

    def _query_hash(self, query: str) -> str:
        norm_q = " ".join(query.strip().split())
        return hashlib.sha256(norm_q.encode("utf-8")).hexdigest()[:24]

    def clean_generated_keywords(self, text: str) -> str:
        """Strips conversational preambles, category headers, bullet points, and parentheticals."""
        if not text or not text.strip():
            return ""
        lines = text.strip().split("\n")
        cleaned_tokens = []
        for l in lines:
            l = l.strip()
            if not l:
                continue
            l_lower = l.lower()
            # Drop conversational filler lines
            if any(l_lower.startswith(p) for p in [
                "here", "the following", "sure", "certainly", "list of", "keywords for", "relevant keywords"
            ]):
                continue
            # Drop category headers: e.g. **Core Concepts**, Primary Terms:
            if re.match(r"^\*{0,2}[A-Za-z\s&/]+:\*{0,2}$", l) or (l.startswith("**") and l.endswith("**") and len(l.split()) <= 4):
                continue
            # Strip bullet points, numbered markers, asterisks, backticks
            l_clean = re.sub(r"^(?:[\*\-\•]|\d+[\.\)])\s*", "", l)
            l_clean = re.sub(r"[\*`_]", "", l_clean)
            # Strip parenthetical explanations: e.g. "Pony (Programming language)" -> "Pony"
            l_clean = re.sub(r"\([^\)]*\)", "", l_clean).strip()
            if l_clean:
                cleaned_tokens.append(l_clean)
        if not cleaned_tokens:
            return text.strip()
        return ", ".join(cleaned_tokens)

    def generate_keywords(self, query: str) -> str:
        """Calls local Ollama with caching, thinking=False, and clean keyword extraction."""
        q_hash = self._query_hash(query)
        cache_file = os.path.join(self.raw_cache_dir, f"{q_hash}.json")
        if not os.path.exists(cache_file):
            try:
                from evaluation.baselines.pyterrier_harness import sanitize_default_query
                sanitized = sanitize_default_query(query)
                if sanitized != query:
                    s_file = os.path.join(self.raw_cache_dir, f"{self._query_hash(sanitized)}.json")
                    if os.path.exists(s_file):
                        cache_file = s_file
            except Exception:
                pass

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("response", "")
            except Exception:
                pass

        prompt = self.prompt_tmpl.format(query=query)
        if not self.think:
            # Dual protection against thinking mode latency (as in legacy pipeline v1):
            # 1. API payload "think": False
            # 2. Prompt injection closing <think></think> directly so model never enters thinking trace
            if "<think>" not in prompt:
                prompt = f"{prompt}\n<think>\n</think>\n"

        payload = {
            "model": self.model_tag,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "keep_alive": -1,  # Keep model warm in RAM across all queries and datasets
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            res_json = resp.json()
            raw_out_text = res_json.get("response", "").strip()
            # Strip any residual <think> tags if present
            if "</think>" in raw_out_text:
                raw_out_text = raw_out_text.split("</think>")[-1].strip()
            elif "<think>" in raw_out_text:
                raw_out_text = raw_out_text.split("<think>")[0].strip()

            out_text = self.clean_generated_keywords(raw_out_text)

            total_duration_ns = res_json.get("total_duration", 0)
            gen_time_ms = round(total_duration_ns / 1e6, 2) if total_duration_ns > 0 else round(wall_ms, 2)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "query": query,
                    "prompt": prompt,
                    "raw_response": raw_out_text,
                    "response": out_text,
                    "gen_time_ms": gen_time_ms,
                    "eval_count": res_json.get("eval_count", 0),
                    "eval_duration_ms": round(res_json.get("eval_duration", 0) / 1e6, 2)
                }, f, indent=2)
            return out_text
        except Exception as e:
            print(f"[LLM-QE Warning] Failed to generate keywords for query '{query[:30]}...': {e}")
            return ""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res_df = df.copy()
        new_query_toks = []

        for idx, row in res_df.iterrows():
            q_text = row["query"]
            orig_terms, _ = self.analyzer.analyze(q_text)
            orig_toks = self.analyzer.to_query_toks(orig_terms)

            # Generate keyword text
            gen_text = self.generate_keywords(q_text)
            gen_terms, _ = self.analyzer.analyze(gen_text)

            if not gen_terms:
                # Exact BM25_Default fallback
                new_query_toks.append(orig_toks)
                continue

            # Q' = Concat(5 copies of Q, G(Q))
            # In analyzed terms: 5 * orig_terms + gen_terms
            combined_terms = (orig_terms * self.repetition_r) + gen_terms
            expanded_toks = self.analyzer.to_query_toks(combined_terms)
            new_query_toks.append(expanded_toks)

        res_df["query_toks"] = new_query_toks
        return res_df
