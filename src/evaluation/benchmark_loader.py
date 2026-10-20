"""
Benchmark loader for the 13 BEIR and 12 BRIGHT collections of the paper.
Streams directly from the raw files under data/raw/ without conversion:
1. BEIR: corpus.jsonl (document text f"{title} {text}".strip() if title else text), queries.jsonl, qrels/test.tsv
   (graded qrels; a query is judged if at least one document has a positive grade). ArguAna: each query's own
   argument is excluded from its ranking (BEIR's ignore_identical_ids).
2. BRIGHT: documents/<domain>-00000-of-00001.parquet and examples/<domain>-00000-of-00001.parquet (gold ids and
   the per-query excluded ids); document ids are sanitised (sanitize_bright_doc_id).
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional, Generator
import numpy as np
import pandas as pd

# Robust project root discovery for data/
_cand_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists(os.path.join(_cand_dir, "data", "raw")):
    BASE_DIR = _cand_dir
elif os.path.exists(os.path.join(os.path.dirname(_cand_dir), "data", "raw")):
    BASE_DIR = os.path.dirname(_cand_dir)
else:
    BASE_DIR = os.getcwd()

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def sanitize_bright_doc_id(raw_id: str) -> str:
    """Sanitize doc IDs with forward/backward slashes, colons, or spaces."""
    return str(raw_id).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


ALL_BRIGHT_DOMAINS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
]


def parse_bright_domain(dataset_name: str) -> Optional[str]:
    """Extracts valid BRIGHT domain name if dataset_name refers to BRIGHT."""
    norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")
    if norm.startswith("bright_"):
        cand = norm[7:]
        if cand in ALL_BRIGHT_DOMAINS:
            return cand
    elif norm in ALL_BRIGHT_DOMAINS:
        return norm
    return None



# BEIR subsets whose queries are corpus documents under the same id (verified 2026-10-11: all 1,298 arguana
# id matches contain the query text; fiqa, fever and climate_fever ids only collide by chance). This follows
# BEIR's ignore_identical_ids for arguana only (BEIR applies it to every dataset by default).
SELF_MATCH_SUBSETS = frozenset({"arguana"})

class BenchmarkLoader:
    """
    Loader of the 25 collections: stream_corpus yields (doc_id, text) without materialising the corpus;
    load_queries returns the judged queries (keys 'query_id', 'question', 'gold_doc_ids', 'qrels', 'excluded_doc_ids')
    and a stats dict; load_query_texts_only returns query_id -> text without touching any label.
    """

    @classmethod
    def _find_beir_dir(cls, subset: str) -> str:
        candidates = [
            os.path.join(RAW_DIR, "beir", subset),
            os.path.join(RAW_DIR, "beir", subset.replace("_", "-")),
            os.path.join(RAW_DIR, "beir", subset.replace("-", "_")),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"BEIR subset '{subset}' not found in {os.path.join(RAW_DIR, 'beir')}")

    @classmethod
    def stream_corpus(cls, dataset_name: str) -> Generator[Tuple[str, str], None, None]:
        """
        Streams (doc_id, text) one-by-one from the raw source file.
        Does NOT materialize the full corpus in RAM.
        """
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")

        bright_domain = parse_bright_domain(dataset_name)
        if bright_domain:
            src_dir = os.path.join(RAW_DIR, "bright")
            doc_parquet = os.path.join(src_dir, "documents", f"{bright_domain}-00000-of-00001.parquet")
            if not os.path.exists(doc_parquet):
                raise FileNotFoundError(f"BRIGHT domain '{bright_domain}' parquet not found at {doc_parquet}")
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(doc_parquet)
            for batch in pf.iter_batches(batch_size=5000):
                pydict = batch.to_pydict()
                ids = pydict.get("id", [])
                contents = pydict.get("content") or pydict.get("text") or []
                for raw_id, content in zip(ids, contents):
                    safe_id = sanitize_bright_doc_id(str(raw_id))
                    text_str = str(content).strip()
                    if safe_id and text_str:
                        yield (safe_id, text_str)
            return

        if norm.startswith("beir_") or norm in (
            "scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "quora",
            "hotpotqa", "fever", "nq", "climate_fever", "dbpedia_entity",
            "trec_covid", "webis_touche2020"
        ):
            subset = norm[5:] if norm.startswith("beir_") else norm
            src_dir = cls._find_beir_dir(subset)
            corpus_file = os.path.join(src_dir, "corpus.jsonl")
            if not os.path.exists(corpus_file):
                corpus_file = os.path.join(src_dir, "corpus_corpus.jsonl")

            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    did = str(rec.get("_id", ""))
                    title = (rec.get("title") or "").strip()
                    text = (rec.get("text") or "").strip()
                    full_text = f"{title} {text}".strip() if title else text
                    if did and full_text:
                        yield (did, full_text)
        else:
            raise ValueError(f"Unknown benchmark dataset: {dataset_name}")

    @classmethod
    def get_corpus_source_paths(cls, dataset_name: str) -> List[str]:
        """
        Returns a list of absolute paths to the raw corpus files consumed by stream_corpus().
        """
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")
        bright_domain = parse_bright_domain(dataset_name)
        if bright_domain:
            src_dir = os.path.join(RAW_DIR, "bright")
            doc_parquet = os.path.join(src_dir, "documents", f"{bright_domain}-00000-of-00001.parquet")
            return [os.path.abspath(doc_parquet)]

        subset = norm[5:] if norm.startswith("beir_") else norm
        src_dir = cls._find_beir_dir(subset)
        c_file = os.path.join(src_dir, "corpus.jsonl")
        if not os.path.exists(c_file):
            c_file = os.path.join(src_dir, "corpus_corpus.jsonl")
        return [os.path.abspath(c_file)]

    @classmethod
    def load_query_texts_only(cls, dataset_name: str) -> Dict[str, str]:
        """
        Loads strictly query IDs and raw text strings without opening, inspecting,
        or touching qrels, exclusions, or relevance labels.
        Guarantees structural label blindness for candidate scoring and inference.
        Returns a dict mapping query_id -> query_text.
        """
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")
        bright_domain = parse_bright_domain(dataset_name)
        if bright_domain:
            src_dir = os.path.join(RAW_DIR, "bright")
            example_parquet = os.path.join(src_dir, "examples", f"{bright_domain}-00000-of-00001.parquet")
            if not os.path.exists(example_parquet):
                raise FileNotFoundError(f"BRIGHT domain {bright_domain} examples parquet not found at {example_parquet}")
            df = pd.read_parquet(example_parquet, columns=["id", "query"])
            queries_map = {}
            for row in df.itertuples(index=False):
                raw_id = str(row.id)
                qtext = str(row.query).strip()
                if raw_id and qtext:
                    queries_map[f"q_bright_{bright_domain}_{raw_id}"] = qtext
            return queries_map

        subset = norm[5:] if norm.startswith("beir_") else norm
        src_dir = cls._find_beir_dir(subset)
        queries_file = os.path.join(src_dir, "queries.jsonl")
        if not os.path.exists(queries_file):
            queries_file = os.path.join(src_dir, "queries_queries.jsonl")
        if not os.path.exists(queries_file):
            raise FileNotFoundError(f"BEIR subset {subset} queries file not found at {queries_file}")

        queries_map = {}
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = str(rec.get("_id", ""))
                qtext = (rec.get("text") or "").strip()
                if qid and qtext:
                    queries_map[f"q_beir_{subset}_{qid}"] = qtext
        return queries_map

    @classmethod
    def load_queries(cls, dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads queries and qrels without materializing the corpus in RAM.
        """
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")

        bright_domain = parse_bright_domain(dataset_name)
        if bright_domain:
            return cls._load_bright_queries(bright_domain)

        if norm.startswith("beir_") or norm in (
            "scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "quora",
            "hotpotqa", "fever", "nq", "climate_fever", "dbpedia_entity",
            "trec_covid", "webis_touche2020"
        ):
            subset = norm[5:] if norm.startswith("beir_") else norm
            src_dir = cls._find_beir_dir(subset)

            queries_file = os.path.join(src_dir, "queries.jsonl")
            if not os.path.exists(queries_file):
                queries_file = os.path.join(src_dir, "queries_queries.jsonl")
            qrels_file = os.path.join(src_dir, "qrels", "test.tsv")

            queries_map = {}
            with open(queries_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    qid = str(rec.get("_id", ""))
                    qtext = (rec.get("text") or "").strip()
                    if qid and qtext:
                        queries_map[qid] = qtext

            qrels_by_q: Dict[str, Dict[str, float]] = {}
            total_qrel_entries = 0
            with open(qrels_file, "r", encoding="utf-8") as f:
                f.readline()  # header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, did, score_val = str(parts[0]), str(parts[1]), float(parts[2])
                        total_qrel_entries += 1
                        qrels_by_q.setdefault(qid, {})[did] = score_val

            formatted_queries = []
            for qid in sorted(queries_map.keys()):
                if qid in qrels_by_q:
                    qrels = qrels_by_q[qid]
                    gold_dids = [did for did, s in qrels.items() if s > 0]
                    # BEIR's ignore_identical_ids, applied only where the document with the query's id is the
                    # query itself (arguana: the query argument is a corpus document); elsewhere ids only collide.
                    self_excl = {qid} if subset in SELF_MATCH_SUBSETS and qid not in qrels else set()
                    if gold_dids:
                        formatted_queries.append({
                            "query_id": f"q_beir_{subset}_{qid}",
                            "question": queries_map[qid],
                            "gold_doc_ids": sorted(gold_dids),
                            "qrels": qrels,
                            "excluded_doc_ids": self_excl,
                            "exclusions": self_excl,
                        })

            stats = {
                "dataset": f"beir_{subset}",
                "queries_loaded": len(formatted_queries),
                "total_qrel_entries": total_qrel_entries,
            }
            return formatted_queries, stats
        else:
            raise ValueError(f"Unknown benchmark dataset: {dataset_name}")

    @staticmethod
    def _load_bright_queries(domain: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        src_dir = os.path.join(RAW_DIR, "bright")
        example_parquet = os.path.join(src_dir, "examples", f"{domain}-00000-of-00001.parquet")
        if not os.path.exists(example_parquet):
            raise FileNotFoundError(f"BRIGHT domain {domain} examples parquet not found at {example_parquet}")

        ex_df = pd.read_parquet(example_parquet)
        formatted_queries = []
        total_qrels = 0

        for q in ex_df.to_dict(orient="records"):
            qid = str(q.get("id", ""))
            qtext = str(q.get("query", "")).strip()

            raw_gold = q.get("gold_ids", [])
            if isinstance(raw_gold, (list, np.ndarray)) or hasattr(raw_gold, "__iter__"):
                gold_list = [str(g) for g in raw_gold if g is not None]
            elif isinstance(raw_gold, str):
                gold_list = [raw_gold]
            else:
                gold_list = []
            safe_gold = [sanitize_bright_doc_id(g) for g in gold_list if str(g).strip()]

            raw_excluded = q.get("excluded_ids", [])
            if isinstance(raw_excluded, (list, np.ndarray)) or hasattr(raw_excluded, "__iter__"):
                excluded_list = [str(e) for e in raw_excluded if e is not None]
            elif isinstance(raw_excluded, str):
                excluded_list = [raw_excluded]
            else:
                excluded_list = []
            safe_excluded = set([sanitize_bright_doc_id(e) for e in excluded_list if str(e).strip()])

            if safe_gold and qtext:
                qrels = {did: 1.0 for did in safe_gold}
                total_qrels += len(qrels)
                formatted_queries.append({
                    "query_id": f"q_bright_{domain}_{qid}",
                    "question": qtext,
                    "gold_doc_ids": sorted(safe_gold),
                    "qrels": qrels,
                    "excluded_doc_ids": safe_excluded,
                    "exclusions": safe_excluded,
                })

        formatted_queries.sort(key=lambda x: x["query_id"])
        stats = {
            "dataset": f"bright_{domain}",
            "queries_loaded": len(formatted_queries),
            "total_qrel_entries": total_qrels,
        }
        return formatted_queries, stats
