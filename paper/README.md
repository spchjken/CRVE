# Paper tables and figures

This folder regenerates every table and figure of the paper from the stored results under `results/`. Nothing is retrieved
or re-encoded: the scripts re-aggregate stored per-query outcomes and summaries, recompute the bootstrap intervals and Holm
tests, cross-check what they compute against the stored summaries and against selected numbers printed in the paper
(reference constants), and stop with a non-zero exit status on any mismatch. Every number printed in Tables 3 to 6, A1 to A5
and S1 to S6 (Supplementary Materials) is in the outputs; each output holds only rows and columns that the paper prints or that a number of the text uses.

```bash
pip install -r requirements.txt     # numpy, pandas, pyarrow, matplotlib (pinned)
python paper/make_all.py            # about 1 minute, about 2 GB of memory
```

`make_all.py` runs the steps below in separate processes, writes everything to `paper/output/` and then compares each file
byte for byte with the committed copy in `paper/expected/` (`--no-check` skips the comparison).

| Step | Writes (paper/output/) |
|---|---|
| `s51.py` | `T3`, `T3b`, `F1_data`, `T4` (`.csv` + `_rounded.csv`); checks the percentile in the caption of Figure 1 |
| `f1_split.py` | `F1_under1M_data`, `F1_over1M_data`, `figures/F1_under1M.pdf`, `figures/F1_over1M.pdf` (reads `F1_data.csv`) |
| `s52_53.py` | `T5`, `T5b`, `T6`, `F2_data`, `F3_data`, `T7`, `T8`, `pool_and_weighting.md`, `checks_s52_53.csv` |
| `f_sens.py` | `F_sens_data`, `figures/F_sens.pdf` (reads `F2_data.csv`, `F3_data.csv`, `T8.csv`) |
| `appendix.py` | `TA1`, `TA2`, `TB1`, `TB2`, `TB3`, `TB4`, `FB1_data`, `TC1`, `TC2` |
| `gate2_twochannel.py` | `T6b` |

Shared code: `common.py` (collection lists and names, rounding, bootstrap interval, figure style). The `lib_*.py` modules hold
the few functions and constants the table scripts take from the experiment code (the stratified bootstrap contrast, the
Holm step-down, the PRF tuning grid and its selection rule, and the readers of the stored per-query outcomes); their result
paths point at the folders under `results/`.

## Conventions

- Every table has `<id>.csv` (unrounded) and `<id>_rounded.csv` (as printed, as fixed-decimal strings that keep trailing
  zeros: 4 decimals for nDCG differences and intervals, 3 for levels, 1 for percent; Holm levels as percent strings such as
  98.75%; a value that rounds to zero prints as 0.0000, never −0.0000; readings are taken from the unrounded values).
- Figures are vector PDFs saved at their exact width (full width 17.0 cm: F_sens; 12.0 cm and 6.0 cm: F1_under1M,
  F1_over1M) with the constrained layout and no creation date, so a rerun with the pinned package versions reproduces them
  byte for byte.
- **CRVE** is the S2 sidecar, two-channel Gate 1, K = 5, w = 0.20, unless marked. **Six-draw mean** (Table 6) is the
  per-query mean of the six random sidecars (seed 42 and seeds 1–5). **Seed 42** marks the sidecar of 20 randomly drawn
  chunks per term used where the paper says randomly drawn chunks. **Capped** means at most 500 queries per collection (the
  tuned-PRF and tuned-CRVE comparisons of Supplementary Section S1).
- All values use the corrected evaluation of four collections (ArguAna, Biology, Earth Science, Stack Overflow; see
  `results/fix_eval/`), except where an item says it was computed before it: F3 / Figure 4b, the pool-size line and the
  paired difference of the lexical profile channel (T3).
- DBPedia-Entity, FEVER and Climate-FEVER keep document identifiers cut at 64 characters (at most 0.9 percent of judged
  pairs). Over 1M documents, the PPMI lists cover only the anchors of the evaluated queries (exact for them).
- **Readings:** Holm rows for which the non-inferiority margin of −0.002 was set in advance (the BM25 rows of T7, the Holm
  rows of TB1 and TB2) read better / non-inferior (lower bound > −0.002) / unclear. Every other interval is read against
  zero: above zero, below zero, or uncertain where it crosses zero.
- Intervals: query bootstrap within collections (corpus-stratified for macro means), 10,000 replicates, seed 42.

### Names used inside the tables

Notes and labels that name where a result comes from use the folder names under `results/` (for example `prf_oracle/`,
`ours_tuned/`). The internal table ids (T4, T5, ...) differ from the paper's numbering; the table below maps them.

## Tables and figures

All sources are under `results/`.

| Paper | Output | Script | Sources | Notes |
|---|---|---|---|---|
| Table 1 | printed by `scripts/table1.py` | `scripts/table1.py` | `collection_stats.csv` (documents); `queries/<c>.txt` (checked against `feature_sidecar/outcomes_<c>.parquet`); `recheck/coverage_documents_with_chunk.csv` (S2 coverage) | Checked against the printed table; `--sidecar-root` recounts the coverage from the S2 sidecars, `--raw` checks the left-out queries of the notes. Not run by `make_all.py`. |
| Table 2 | (settings) | – | – | Method settings; listed with their place in the code in the top-level `README.md`. |
| Figure 1 | printed by `scripts/figure1_example.py`; caption percentile by `s51.py` | `scripts/figure1_example.py`, `s51.py` | fiqa index, stored Gate 1 assets and S2 sidecar (`data/README.md`); for the caption `feature_sidecar/outcomes_<c>.parquet`, `full_queries/queries.csv` | Worked FiQA example recomputed (Gate 1 channels and RRF, Gate 2 cosines, closest chunks, live BM25); every number of the figure is checked (not run by `make_all.py`). Caption: among the 792 held-out queries of the six BEIR collections under 1M with held-out queries (SciFact, NFCorpus, FiQA, ArguAna, SCIDOCS, Quora) that CRVE (S2) improves over BM25, 46.7% have a smaller gain than the example (47th percentile; checked by `s51.py`). |
| Figure 2 | `figures/F1_under1M.pdf`, `F1_under1M_data.csv` | `f1_split.py` (after `s51.py`) | `F1_data.csv` (from `feature_sidecar/outcomes_<c>.parquet`); `collection_stats.csv` | Collections ordered by number of documents, ascending (ties by name); the tick label carries the count. The written values must equal `F1_data.csv` exactly. Short tick names at 12 cm: TheoremQA-Thm., TheoremQA-Q., Sust. Living. TheoremQA-Theorems over DPH has base 0, so its relative gain is undefined. |
| Figure 3 | `figures/F1_over1M.pdf`, `F1_over1M_data.csv` | `f1_split.py` | as Figure 2 | FEVER (5,416,568) before Climate-FEVER (5,416,593). |
| Table 3 | `T4.csv` (nDCG@10 columns) | `s51.py` | `paper_posthoc/a_holm.json`, `a_contrasts.csv`, `prf_comparison/per_query_<c>.parquet`, `feature_sidecar/outcomes_<c>.parquet` | CRVE minus RM3, Bo1, BGE-QE, LLM-Q2E over BM25/DPH, under/over 1M, all queries; Holm over 8 per group. Relative difference = difference ÷ the baseline's level of the same metric. |
| Table 4 | `T5.csv` | `s52_53.py` | `paper_posthoc/a_risk_profile.csv`, `a_robustness_index.csv`, `prf_comparison/per_query_<c>.parquet`, `prf_oracle/per_query_arms.parquet`, `prf_oracle/risk_profile.csv` | Risk per method × base × group (the 20 rows of the table, all queries). Each method is compared with its own base (baselines against the harness base; CRVE against our base). Also written: the counts of tuned RM3 and Bo1 (best-of-both oracle, cross-validated) on the capped sets, which Section 5.2 quotes (marked capped; against our base). |
| Table 5 | `T6b.csv` | `gate2_twochannel.py` | `gate2_twochannel/summary.csv`, `outcomes_<c>.parquet` | Full method (Gate 2 top K) and Gate 1 order alone (first K of the same 200 candidates), each against the base, and the paired contribution of Gate 2; K = 5, 10; all queries, S2. Recomputed from the per-query outcomes and checked against `summary.csv`. |
| Figure 4 | `figures/F_sens.pdf`, `F_sens_data.csv` | `f_sens.py` (after `s52_53.py`) | `F2_data.csv`, `F3_data.csv`, `T8.csv` | (a) Gate 1 depth L, under 1M (from F2: `gate1_depth/`); (b) chunk grid at K = 5 (from F3: `chunk_grid/surface.csv`, `query_outcomes_{dev,new}.parquet`; **predates the evaluation fixes**, random chunks, five-channel Gate 1, 18 collections under 1M); (c) K × w (from T8). `F_sens_data.csv` holds every value drawn. |
| Table 6 | `T7.csv` | `s52_53.py` | `feature_sidecar/analysis*.json`, `summary*.csv`, `outcomes_<c>.parquet`; `s4/primary_holm.json`, `summary*.csv`; six draws from `draws/`, `dph/`, `pool_size/`, `pool_reduced/`, `fix_eval/`, `full_queries/outcomes.parquet`, `gate1_channels/outcomes.parquet` | Chunk-selection rules S2, S1, S4 minus the six-draw mean; BM25 rows Holm, DPH rows 95%. The six draws' range of gains is recomputed from the stored draw outcomes. |
| Table A1 | `TA1.csv` | `appendix.py` | `prf_comparison/per_query_<c>.parquet`, `feature_sidecar/outcomes_<c>.parquet` | Per-collection nDCG@10. Our base and the harness base differ by at most 0.0034 (Sustainable Living, BM25). |
| Table A2 | `TA2.csv` | `appendix.py` | as Table A1 | Macro nDCG@10/20/50, Recall@100/1000. |
| Table A3 | `T4.csv` (nDCG@50, Recall@1000 columns) | `s51.py` | as Table 3 | 95% intervals. |
| Table A4 | `T8.csv` | `s52_53.py` | `recheck/risk_by_cell.csv`, `ours_tuned/grid_<c>.parquet` (base levels) | Capped sets, so the default cell is +0.0091 here against +0.0090 in Table 5. |
| Table A5 | `T6.csv` | `s52_53.py` | `recheck/gate1_order_paired.csv`, `gate1_order_18_corpora.csv`, `recheck/{unseen_queries,new_corpora}/query_outcomes_F.parquet`, `queries.csv` | Gate 2 against the five-channel Gate 1 order (randomly drawn chunks, BM25), 18 collections. |
| Table S1 | `TB1.csv` | `appendix.py` | `ours_tuned/per_query_arms.parquet`, `posthoc_cv_gain_over_default.csv`; `prf_oracle/grid_<c>.parquet`, `per_query_arms.parquet` | Capped sets. Cross-validated PRF recomputed for seeds 0–29 and 42. |
| Table S2 | `TB2.csv` | `appendix.py` | `ours_tuned/{analysis.json,summary.csv,cv_split_sensitivity.csv}` | Both sides tuned. |
| Table S3 | `TB3.csv` | `appendix.py` | `ours_tuned/{analysis.json,summary.csv,posthoc_cv_gain_over_default.csv,per_query_arms.parquet}` | Tuning headroom; RM3/Bo1 CV − default intervals recomputed. |
| Table S4 | `TB4.csv` | `appendix.py` | `ours_tuned/risk_profile.csv`, `per_query_arms.parquet` | Risk counts of the tuned arms. |
| Table S5 | `TC1.csv` | `appendix.py` | `cost/build_costs.csv` | Build cost per collection. Costs the random seed-42 sidecar. Over-1M values are lower bounds (restricted PPMI lists; their peak memory not measured). |
| Table S6 | `TC2.csv` | `appendix.py` | `cost/latency_summary.csv`, `cost/latency_<c>_{base,opt}.parquet` | Latency per stage (median, p90), five collections. The GPU-encoding rows come from a separate timing session (`latency_<c>_opt.parquet`). Random seed-42 sidecar in the benchmark. |

Outputs that are not tables of the paper but hold numbers of its text: `T3` (Gate 1 channels, Section 5.3), `T3b`
(collections above/below ±0.005, Section 5.1), `F1_data` (per-collection gains, Sections 1 and 5.1), `T5b` (Quora by BM25
level, Section 6.2), `FB1_data` (per collection, CRVE minus the best-of-both PRF oracle; Sections 6.2, 6.3 and Supplementary Section S1),
`pool_and_weighting.md` (Section 5.3), `F2_data`, `F3_data` (Figure 4a, 4b), `checks_s52_53.csv` (the cross-checks of
`s52_53.py`).

## Numbers in the text

| Section | Number | Source |
|---|---|---|
| Abstract, 1, 5.1 | Gain over BM25 / DPH under and over 1M (+0.0090, +0.0112; +0.0053, DPH uncertain) and relative gains | `T6b.csv` (K = 5) |
| Abstract, 1, 5.2, 7 | Collections lowered by more than 0.005 (one of 20; 7 to 16 for the baselines) | `T5.csv` |
| Abstract, 1, 5.1 | Ahead of RM3 / Bo1 / BGE-QE by up to 13.1% and 7.0%; LLM-Q2E comparison | `T4.csv` |
| 1, 5.1 | Per-collection gains (NFCorpus +0.035, intervals above zero on 9 / 10 of 20, over-1M collections) | `F1_data.csv` |
| 1, 6.2 | Per-query time relative to LLM-Q2E (12% to 15%; 6.6 to 8.6 times) | `results/cost/latency_summary.csv` |
| 1, 5.3 | Gate 1 order alone vs Gate 2 (+0.0086 of +0.0090; +0.0093 of +0.0112; +0.0059 over 1M) | `T6b.csv` |
| 1, 6.1, 7 | Greedy oracle top 10: +0.0095 (+4.0%) to +0.0258 (+10.7%); share of queries with a helpful term; within-query AUC | stored summaries only (no per-query outcomes are shipped, so these numbers are not recomputed here): `results/oracle_top10/headroom_ratio.csv`, `decomposition_o2.csv`, `analysis.json`, `posthoc_auc_sensitivity.csv` |
| 4 | 25 collections, 13 BEIR and 12 BRIGHT; document counts | `results/collection_stats.csv`, `results/recheck/coverage_documents_with_chunk.csv` |
| 4 | 33,624 judged queries (14,166 under 1M, 19,458 over) | `results/feature_sidecar/outcomes_<c>.parquet`; `F1_data.csv` (queries) |
| 4 | Two BM25 base runs differ by at most 0.0014 (Sustainable Living 0.0034); DPH | `TA1.csv` |
| 4 | 1,559 of 33,624 queries longer than 128 tokens | `results/paper_posthoc/d_query_truncation.csv`, `run_d.json` |
| 4 | Quora holds 70% of the queries under 1M | `F1_data.csv` (queries) |
| 5.1 | nDCG@50 and Recall@1000 readings | `T4.csv` |
| 5.1, Suppl. S1 | Against tuned PRF (cross-validated and oracle); best-of-both −0.0074 | `TB1.csv` |
| 5.2 | Tuned RM3/Bo1 risk counts (11 and 3; 5 and 3; 8 to 13) | `TB4.csv`; `T5.csv` (capped rows) |
| 5.2 | Robustness index (mean +0.074; pooled −0.037 / −0.068) | `T5.csv` |
| 5.2, 6.3 | Quora and other per-collection levels | `TA1.csv` |
| 5.3 | Gate 1 channels on 30,541 queries (+0.0115, +0.0039; co-occurrence channel alone +0.0096, +0.0037) | `T3.csv` (gain rows) |
| 5.3 | Paired difference from adding the lexical profile channel, before the evaluation corrections: uncertain in both groups | `T3.csv` (paired rows; recomputed from `results/full_queries/outcomes.parquet`, checked against `results/full_queries/summary.csv`) |
| 5.3 | Gate 2 over the five-channel order: +0.0056 [+0.0028, +0.0084] | `T6.csv` |
| 5.3 | Gate 1 depth L curve | `F2_data.csv` |
| 5.3 | Chunk-selection rules (S2 non-inferior; DPH +0.0023) | `T7.csv` |
| 5.3 | Risk grows with w; K effects | `T8.csv` |
| 5.3 | Chunks per term and chunk length near a plateau | `F3_data.csv` |
| 5.3 | Pool size (30% pool keeps about 85% of the gain; 66% to 103%; FiQA, Sustainable Living, ArguAna 36% to 49%) and per-term weighting (+0.0005, −0.0008) | `pool_and_weighting.md` (computed from `results/pool_reduced/rep_verdict.json`, `rep_extra.json` and `results/recheck/full_queries/verdict.json`) |
| 6.1 | Gain by sidecar coverage (share > 1/3: 13 collections) | `results/paper_posthoc/b_gain_by_share.csv`, `b_shares.csv` (from `results/chunk_counts/`) |
| 6.1 | 3% to 6% of documents with a stored chunk over 1M | `results/recheck/coverage_documents_with_chunk.csv` |
| 6.1 | Median added term in less than 0.5% of documents | `results/paper_posthoc/c_df_of_chosen_terms.csv` |
| 6.1 | Quora: CRVE's added terms about eight times rarer than LLM-Q2E's | `results/paper_posthoc/c_df_of_chosen_terms.csv`, `results/added_term_profile/summary.json` |
| 6.2, Suppl. S1 | CRVE vs best-of-both PRF oracle per collection (10/9/6, 9/10/6) | `FB1_data.csv` |
| 6.2 | Quora: BM25 perfect on 60% of queries; about four fifths of CRVE's net loss from them | `T5b.csv` (from `results/added_term_profile/ceiling_by_bm25_level.csv`) |
| 6.3 | TREC-COVID: the best of tuned RM3 and Bo1 ahead by 0.010 (−1.5%) | `FB1_data.csv` |
| 6.3 | Build takes minutes under and hours over 1M | `TC1.csv` |
| Suppl. S1 | Grids and chosen settings | `results/ours_tuned/chosen_cells.csv`, `results/prf_oracle/chosen_configs.csv` |
| Suppl. S1 | Both sides tuned (14 of 16), tuning headroom | `TB2.csv`, `TB3.csv` |
| Suppl. S2 | Retrieval with five added terms costs about 5 to 17 ms more; LLM-Q2E 604 to 757 ms; fresh generations differ from the cache on 44% of timed queries | `TC2.csv`, `results/cost/latency_summary.csv`, `results/cost/run_latency_<c>_llm.json` |
| Suppl. S2 | CPU run of the GPU session 78–103 ms; same top five terms on CPU and GPU | `TC2.csv`; `results/cost/run_latency_<c>_opt.json` (`same_top5`) |

Sources named without a folder are outputs in `paper/output/`.
