"""
Terrier baseline harness: BM25 and DPH (Terrier defaults), RM3 and Bo1 (Terrier implementations with 10 feedback
documents, 10 expansion terms and, for RM3, lambda = 0.5, set explicitly), and the BGE-QE / LLM-Q2E rewriters followed by
BM25 or DPH.

- The harness saves each run; the per-query metrics of the paper are recomputed from the saved runs
  (crve.pipeline.baselines.per_query_from_runs). It prints the mean linear-gain nDCG@10 and Recall@1000 (ir_measures).
- BRIGHT exclusions: excluded documents are removed before and after feedback, with adaptive depth padding.
- Queries are processed in chunks of 200; runs can be saved as Parquet (qid, docno, score, rank).

Indexes are built by scripts/build_index.py with the settings frozen in configs/index_settings.yaml.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Set
import pandas as pd

import pyterrier as pt
import ir_measures
from ir_measures import nDCG, R

# printed summary measures (linear gains)
MEASURES = [nDCG @ 10, R @ 1000]


def init_pyterrier(mem: int = 3072):
    """Idempotent PyTerrier initialization with bounded JVM heap and OOM protection."""
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
                f.write("500\n")
        except Exception:
            pass

    if not pt.java.started():
        try:
            pt.java.set_memory_limit(mem)
        except Exception:
            pass
        pt.java.init()


def sanitize_default_query(query_text: str) -> str:
    """
    Sanitizes raw query text for standard TerrierQL retrieval.
    Removes syntax characters (?, :, <, >, {, }, ^, $, etc.) that trigger
    QueryParserException, matching EnglishTokeniser document-side behavior.
    """
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(query_text))
    tokens = [w for w in cleaned.split() if w.strip()]
    return " ".join(tokens) if tokens else "a"


def create_exclusion_filter(excluded_map: Optional[Dict[str, Set[str]]] = None, max_docs: Optional[int] = None):
    """
    Creates a PyTerrier transformer that drops documents in excluded_map[qid],
    slices each query to top max_docs, and recomputes contiguous 0-indexed ranks.
    Preserves all PyTerrier metadata columns (qid, query, docno, score, docid, etc.).
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        
        if excluded_map:
            mask = [
                str(row["docno"]) not in excluded_map.get(str(row["qid"]), set())
                for _, row in df.iterrows()
            ]
            df = df[mask]
        
        if max_docs is not None:
            df = df.groupby("qid").head(max_docs)
            
        df = df.reset_index(drop=True)
        df["rank"] = df.groupby("qid").cumcount()
        return df

    return pt.apply.generic(_filter)


class PyTerrierBaselineHarness:
    """
    The baseline pipelines over one Terrier index (index_dict["index_default"]):
      BM25_Default, DPH                    BM25 and DPH
      BM25_RM3, BM25_Bo1, DPH_RM3, DPH_Bo1 RM3 / Bo1 (10 feedback documents, 10 terms, RM3 lambda = 0.5)
      BGE_Vocab_QE, DPH_BGE_Vocab_QE       BGE-QE rewriter (bge_vocab_rewriter) followed by BM25 / DPH
      LLM_Q2E_ZS, DPH_LLM_Q2E_ZS           LLM-Q2E rewriter (llm_qe_rewriter) followed by BM25 / DPH
    """

    def __init__(self, index_dict: Optional[Dict[str, Any]] = None):
        index_dict = index_dict or {}
        self.index_default = index_dict.get("index_default")

    def warmup(self, num_queries: int = 10):
        """Warm up JVM JIT compiler with dummy queries before measurement."""
        if self.index_default is None:
            return
        dummy_q = [{"qid": f"warmup_{i}", "query": "retrieval search machine learning algorithm"} for i in range(num_queries)]
        df_dummy = pd.DataFrame(dummy_q)
        try:
            bm25 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=10)
            _ = bm25.transform(df_dummy)
        except Exception:
            pass

    def build_pipeline(self, pipeline_name: str, excluded_map: Optional[Dict[str, Set[str]]] = None) -> Any:
        """
        Builds the specified retrieval pipeline enforcing strict BRIGHT pre/post exclusion filtering.
        """
        max_ex = max([len(s) for s in excluded_map.values()] or [0]) if excluded_map else 0

        if pipeline_name == "BM25_Default":
            if excluded_map:
                retriever = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(1000 + max_ex, 3000))
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retriever >> filter_post
            return pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=1000)

        elif pipeline_name == "DPH":
            if excluded_map:
                retriever = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(1000 + max_ex, 3000))
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retriever >> filter_post
            return pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=1000)

        elif pipeline_name == "BM25_RM3":
            if excluded_map:
                # First pass: request K1 = min(max(100, fb_docs + max_ex), 300), filter excluded, slice to fb_docs=10
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                # Second pass: request K2 = min(1000 + max_ex, 3000), filter excluded, slice to 1000
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> rm3 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=1000)
                return pass1 >> rm3 >> pass2

        elif pipeline_name == "BM25_Bo1":
            if excluded_map:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> bo1 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=1000)
                return pass1 >> bo1 >> pass2

        elif pipeline_name == "DPH_Bo1":
            if excluded_map:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> bo1 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=1000)
                return pass1 >> bo1 >> pass2

        elif pipeline_name == "DPH_RM3":
            if excluded_map:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> rm3 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=1000)
                return pass1 >> rm3 >> pass2

        elif pipeline_name == "BGE_Vocab_QE":
            if not hasattr(self, "bge_vocab_rewriter") or self.bge_vocab_rewriter is None:
                raise ValueError("BGE_Vocab_QE requires bge_vocab_rewriter to be registered on harness")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.bge_vocab_rewriter >> pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif pipeline_name == "DPH_BGE_Vocab_QE":
            if not hasattr(self, "bge_vocab_rewriter") or self.bge_vocab_rewriter is None:
                raise ValueError("DPH_BGE_Vocab_QE requires bge_vocab_rewriter to be registered on harness")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.bge_vocab_rewriter >> pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif pipeline_name == "LLM_Q2E_ZS":
            if not hasattr(self, "llm_qe_rewriter") or self.llm_qe_rewriter is None:
                raise ValueError("LLM_Q2E_ZS requires llm_qe_rewriter to be registered on harness")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.llm_qe_rewriter >> pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif pipeline_name == "DPH_LLM_Q2E_ZS":
            if not hasattr(self, "llm_qe_rewriter") or self.llm_qe_rewriter is None:
                raise ValueError("DPH_LLM_Q2E_ZS requires llm_qe_rewriter to be registered on harness")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.llm_qe_rewriter >> pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

    def evaluate_pipeline(
        self,
        pipeline_name: str,
        queries: List[Dict[str, Any]],
        qrels_ir: List[ir_measures.Qrel],
        gold_map: Dict[str, Dict[str, float]],
        chunk_size: int = 200,
        save_runs_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the pipeline over the queries in chunks of chunk_size, saves the run to
        <save_runs_dir>/<dataset>_<pipeline>.parquet (qid, docno, score, rank) and returns the mean nDCG@10 and Recall@1000.
        """
        # 1. Query formatting & exclusion extraction
        excluded_map = {str(q["query_id"]): set(q.get("excluded_doc_ids", [])) for q in queries if q.get("excluded_doc_ids")}
        formatted_queries = [
            {"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])}
            for q in queries
        ]

        # 2. Build Pipeline
        transformer = self.build_pipeline(pipeline_name, excluded_map=excluded_map)

        # 3. Retrieve and accumulate the measures in bounded query chunks
        total_queries = len(queries)
        metric_sums = {m: 0.0 for m in MEASURES}

        # Pre-group qrels by qid for fast chunk lookup
        qrels_by_qid: Dict[str, List[ir_measures.Qrel]] = {}
        for qrel in qrels_ir:
            qrels_by_qid.setdefault(str(qrel.query_id), []).append(qrel)

        persisted_chunks = []

        for i in range(0, total_queries, chunk_size):
            chunk_queries = queries[i : i + chunk_size]
            chunk_df_q = pd.DataFrame(formatted_queries[i : i + chunk_size])
            res_c = transformer.transform(chunk_df_q)

            chunk_run_ir = []
            for _, row in res_c.iterrows():
                qid = str(row["qid"])
                docno = str(row["docno"])
                # Defensive check against exclusions
                if qid in excluded_map and docno in excluded_map[qid]:
                    continue
                chunk_run_ir.append(ir_measures.ScoredDoc(qid, docno, float(row["score"])))

            chunk_qrels = []
            for q in chunk_queries:
                chunk_qrels.extend(qrels_by_qid.get(str(q["query_id"]), []))
            if chunk_run_ir and chunk_qrels:
                for mv in ir_measures.iter_calc(MEASURES, chunk_qrels, chunk_run_ir):
                    metric_sums[mv.measure] += float(mv.value)

            if save_runs_dir and dataset_name:
                persisted_chunks.append(res_c[["qid", "docno", "score", "rank"]].copy())

            del res_c, chunk_run_ir, chunk_qrels

            if total_queries > 500 and (min(i + chunk_size, total_queries) % 1000 < chunk_size or (i + chunk_size) >= total_queries):
                print(f"  [{pipeline_name}] Progress: {min(i + chunk_size, total_queries)}/{total_queries} queries evaluated...", flush=True)

        # Persist the run as Parquet
        if save_runs_dir and dataset_name and persisted_chunks:
            os.makedirs(save_runs_dir, exist_ok=True)
            safe_ds = dataset_name.lower().replace("-", "_")
            parquet_path = os.path.join(save_runs_dir, f"{safe_ds}_{pipeline_name}.parquet")
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                writer = None
                for chunk_df in persisted_chunks:
                    table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
                    writer.write_table(table)
                if writer is not None:
                    writer.close()
            except Exception:
                run_df = pd.concat(persisted_chunks, ignore_index=True)
                run_df.to_parquet(parquet_path, compression="snappy", index=False)
                del run_df
            print(f"  [{pipeline_name}] Persisted candidate run -> {parquet_path}", flush=True)
            del persisted_chunks

        total_q = max(1, total_queries)
        return {
            "pipeline": pipeline_name,
            "ndcg_10": round(metric_sums[MEASURES[0]] / total_q, 4),
            "recall_1000": round(metric_sums[MEASURES[1]] / total_q, 4),
        }

    def run_all(
        self,
        queries: List[Dict[str, Any]],
        chunk_size: int = 200,
        save_runs_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        pipelines: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Runs and evaluates the specified baselines across the query set."""
        # Convert queries and gold qrels to ir_measures Qrel list
        qrels_ir = []
        gold_map = {}
        for q in queries:
            qid = str(q["query_id"])
            golds = q.get("qrels") or {str(did): 1.0 for did in q.get("gold_doc_ids", [])}
            gold_map[qid] = golds
            for doc_id, rel in golds.items():
                qrels_ir.append(ir_measures.Qrel(qid, str(doc_id), int(rel)))

        # Warm up
        self.warmup(num_queries=10)

        if pipelines is None:
            pipelines = ["BM25_Default", "BM25_RM3", "BM25_Bo1", "DPH"]

        results = []
        for p in pipelines:
            print(f"[PyTerrier] Evaluating {p}...", flush=True)
            res_dict = self.evaluate_pipeline(
                p,
                queries,
                qrels_ir,
                gold_map,
                chunk_size=chunk_size,
                save_runs_dir=save_runs_dir,
                dataset_name=dataset_name,
            )
            results.append(res_dict)
            import gc
            gc.collect()

        return pd.DataFrame(results)
