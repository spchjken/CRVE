# Provenance of the stored results

The folders under `results/` hold the stored outputs of the experiments behind the paper. The scripts in `paper/` only
re-aggregate them. Per-query files carry nDCG@10/20/30/50 and Recall@50/100/200/500/1000 per collection, query, base model
(BM25 or DPH, Terrier, 1,000 documents per query) and arm. Unless a folder says otherwise, CRVE is run as in the paper: pool of
10,000 terms, two-channel Gate 1 (reciprocal-rank fusion, k = 60, of the PPMI and sparse lexical-profile channels, L = 200
candidates), Gate 2 top K = 5 by maximum cosine between the query and the stored chunks (BGE-small-en-v1.5), added terms with
weight w = 0.20, the NoOp term pipeline, and the common query set of the paper (all judged queries of each collection used
throughout; `results/queries/<c>.txt`). All values use the corrected evaluation of four collections (see `fix_eval/`) unless
marked. Stored files can hold more arms or rows than the paper reports (for example random-sidecar arms); the scripts in
`paper/` read only what the paper uses.

The experiment scripts that produced these files are not part of this release. The code that computes CRVE (S2) and the
baselines is in `src/` (run by `scripts/run_crve.py`, `scripts/run_baselines.py` and `scripts/run_qe_baselines.py`); it
reproduces the per-query values of `feature_sidecar/` (arms S2 and base), `prf_comparison/` and `prf_oracle/` (see the
top-level `README.md`).

## Main results

| Folder | How it was produced |
|---|---|
| `feature_sidecar/` | CRVE with the deterministic chunk-selection sidecars S1 (cheap-vector centrality) and S2 (lowest mean IDF density), 20 chunks per term, on the common query set of all 25 collections, over BM25 and DPH (`outcomes_<c>.parquet`); `analysis*.json` and `summary*.csv` hold the Holm tests of each rule against the six-draw mean and the 95% contrasts. |
| `prf_comparison/` | Per-query metrics of every compared method on the common query set: our BM25/DPH base runs, the harness base runs, RM3 and Bo1 (Terrier implementations with 10 feedback documents, 10 terms and RM3 λ = 0.5, set explicitly), BGE-QE, LLM-Q2E (qwen3.5:4b, cached generations replayed); the files also hold CRVE's six-draw mean (`<base>+ours`, not used). |
| `paper_posthoc/` | Post hoc analyses on stored outcomes: (a) CRVE (S2) against each baseline with Holm over eight contrasts per group, risk profiles and robustness index; (b) gain by the share of chunks stored (from `chunk_counts/`); (c) document frequency of the chosen terms; (d) queries longer than 128 tokens. Run records `run_<stage>.json`. |
| `recheck/` | Analyses recomputed after the evaluation corrections: the 30,541 queries of eleven BEIR collections of Section 5.3 (`full_queries/`: per-query outcomes and the tests of the two-channel Gate 1 and of the co-occurrence channel alone against BM25 and of per-term weighting, randomly drawn chunks); Gate 2 against the five-channel Gate 1 order on 18 collections under 1M (`gate1_order_*.csv`, from the per-query outcomes in `unseen_queries/` and `new_corpora/`, randomly drawn chunks); risk by K × w cell over the tuned grid (`risk_by_cell.csv`); documents with at least one stored chunk (`coverage_documents_with_chunk.csv`). |
| `full_queries/` | The same 30,541 queries before the evaluation corrections: the query list with roles (`queries.csv`; role `fresh` marks the held-out queries), per-query outcomes of BM25 and the expansion arms (`outcomes.parquet`; the over-1M BM25 runs of the seed-42 draw come from here) and their 95% contrasts (`summary.csv`; the paired difference of the lexical profile channel in Section 5.3). |
| `prf_oracle/` | RM3 and Bo1 over BM25 and DPH on a grid of feedback documents (5, 10, 20), feedback terms (10, 20) and, for RM3, λ (0.3, 0.5, 0.7), on at most 500 queries per collection (seed-42 sample); per collection the oracle (best grid point on the evaluated queries) and a two-fold cross-validated choice (`grid_<c>.parquet`, `per_query_arms.parquet`, `risk_profile.csv`, `chosen_configs.csv`). |
| `ours_tuned/` | CRVE S1 and S2 tuned on the same capped query sets over K ∈ {1, 3, 5, 10} × w ∈ {0.10, 0.20, 0.30} (oracle and cross-validated), next to the tuned PRF arms of `prf_oracle/`; `grid_<c>.parquet` holds every cell, `per_query_arms.parquet` the compared arms. |
| `cost/` | Build cost per collection (`build_costs.csv`, from the build records of the pool, PPMI, lexical-profile and Gate 2 sidecars; over 1M a lower bound) and per-query latency on five collections, 100 queries each: session `base` (BM25, RM3, Bo1, BGE-QE, CRVE with CPU query encoding), session `llm` (LLM-Q2E with fresh generation through Ollama), session `opt` (CRVE with CPU and GPU query encoding). `latency_summary.csv` holds the medians and 90th percentiles; `run_*.json` are the run records. |
| `chunk_counts/` | Per collection: chunks, chunks containing a pool term, and unique chunks stored by the random sidecars (seed 42 and seeds 1–5); term coverage. |
| `added_term_profile/` | Quora: document frequency and count of the terms added by LLM-Q2E and by CRVE (S2) (`summary.json`); changes by BM25 level (`ceiling_by_bm25_level.csv`). |
| `collection_stats.csv` | Frozen per-collection document counts (`num.Documents` of each Terrier index), raw corpus size (BEIR: size of `corpus.jsonl`; BRIGHT: UTF-8 bytes of the document text) and Terrier index size on disk, read by `paper/f1_split.py`. |

## Side experiments and inputs to the six-draw mean

For these folders: outcomes only; the generating experiment scripts are not part of this release.

| Folder | How it was produced |
|---|---|
| `draws/` | Six draws: the random Gate 2 sidecar (20 chunks per term) redrawn with seeds 1–5, with (`both`) and without (`g2only`) redrawing the lexical profiles; BM25 and DPH. The seed-42 draw comes from `dph/`, `pool_size/`, `pool_reduced/` and `full_queries/`. |
| `dph/` | The seed-42 draw over DPH (and the DPH base) on the common query set. |
| `pool_size/` | BM25 outcomes of the seed-42 draw on the 20 collections under 1M, for pool fractions α and candidate orders; the paper uses the full pool (α = 1.0, salience order) and the BM25 base. Predates the evaluation corrections. |
| `pool_reduced/` | Reduced pool: the α = 0.3 pool rebuilt in full, five builds, against five builds of the full pool (`repoutcomes_<c>.parquet`); `rep_verdict.json` and `rep_extra.json` hold the kept share of the gain. Predates the evaluation corrections. |
| `fix_eval/` | The four collections re-evaluated after the evaluation corrections (ArguAna, Biology, Earth Science, Stack Overflow): base, seed-42 draw (stored arm label `today`) and the six draws. |
| `gate1_channels/` | Gate 1 channel variants over BM25; over 1M the fusion arm is the seed-42 draw used in the six-draw mean. |
| `gate1_depth/` | Gate 1 depth: L ∈ {10, 25, 50, 100, 200, 300, 500} candidates passed to Gate 2, S1 and S2 sidecars, BM25 and DPH, collections under 1M (Figure 4a). |
| `chunk_grid/` | Chunk grid: chunk length (32 to 256 tokens) × chunks kept per term (5 to 100) × K, random chunks (seed 42), five-channel Gate 1, 18 collections under 1M (Figure 4b). Predates the evaluation corrections. |
| `s4/` | Chunk-selection rule S4 (reciprocal-rank fusion of the S1 and S2 chunk orders) against the six-draw mean (Table 6). |
| `gate2_twochannel/` | Gate 2 against the order of the two-channel Gate 1 list (first K candidates, no Gate 2), K = 5 and 10, BM25 and DPH (Table 5): per-query outcomes (`outcomes_<c>.parquet`; arm A: Gate 2 over S2, arm B: Gate 1 order) and their stored summary (`summary.csv`). |
| `oracle_top10/` | Greedy oracle choice of the expansion terms among the Gate 2 top 10 (an upper bound using the qrels of the evaluated queries; BM25 only; at most 200 queries per collection); headroom ratio and decomposition. Stored summaries only (no per-query outcomes). |

## Changes made for the release

**Result files.** File names inside each folder are unchanged; the folders were renamed after what they hold
(for example `prf_oracle/`, `ours_tuned/`). Private commit hashes and working-tree flags were removed from these run records
(no script reads them; all other fields and the formatting are unchanged):

- `cost/run_build_costs.json` and the 15 `cost/run_latency_<collection>_{base,llm,opt}.json`: field `git_commit_at_start`.
- `paper_posthoc/run_{a,b,c,d}.json`: fields `git_commit`, `crve_dirty`.
- `added_term_profile/summary.json`: field `_meta.git_commit`.

Only the files that a script in `paper/` or `scripts/` reads, or that a number of the paper's text comes from, are shipped.

**Scripts.** `common.py`, `s51.py`, `s52_53.py`, `appendix.py`, `f_sens.py`, `f1_split.py` and `gate2_twochannel.py` are the
table scripts of the paper with paths moved to this layout (inputs under `results/`, outputs under `paper/output/`), imports
pointed at the `lib_*.py` modules, and their outputs restricted to the rows and columns that the paper prints or that a
number of its text uses, with labels in the paper's terms. `f1_split.py` reads the frozen `results/collection_stats.csv`
instead of the Terrier indexes and raw corpora, and checks the document counts against the counts printed in Table 1 (kept as
constants). Reference constants of the cross-checks are numbers printed in the paper; each check names the table or
section.

**Outputs.** Every number printed in Tables 3 to 6 and A1 to A11 is in the outputs (checked cell by cell for Tables 3, 4, 5,
6, A3 and A4, and as a multiset of the printed numbers for the other tables), and the three figure PDFs (Figures 2, 3 and 4)
are byte-identical to the figures of the paper's build.
