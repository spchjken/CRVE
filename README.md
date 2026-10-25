# CRVE: Context-Ranked Vocabulary Expansion

Code, stored results and prebuilt assets for the paper

> **CRVE: Low-Risk, LLM-Free Query Expansion for Lexical Retrieval from Stored Corpus Contexts**
> Dong Hoang Van, Jialu Shen, Md Monjurul Karim, Qiang Qu

Query expansion can improve lexical retrieval, but selecting terms that actually help a query remains difficult.
Context-Ranked Vocabulary Expansion (CRVE) uses a small set of passages stored from the target collection as evidence
for this selection. Offline, it builds a query-independent pool of potentially useful terms from collection statistics
and stores a small set of passages (chunks) for each term, with their encoder vectors (the context sidecar), together
with co-occurrence lists and lexical profiles. At query time, Gate 1 proposes candidate terms from the co-occurrence and
profile channels, Gate 2 ranks them by the similarity between the query and their stored contexts, and the top K are
added to the query at a small fixed weight for a single BM25 or DPH retrieval. No first-pass retrieval and no text
generation are needed. The paper evaluates CRVE over BM25 and DPH on 25 collections from BEIR and BRIGHT.

## Supplementary Materials

The Supplementary Materials of the paper are in [`supplementary/CRVE_supplementary.pdf`](supplementary/CRVE_supplementary.pdf)
(LaTeX source: `supplementary/CRVE_supplementary.tex`):

- Section S1, tuning on both sides: RM3, Bo1 and CRVE tuned per collection (Tables S1 to S4).
- Section S2, cost of the research implementation: offline build cost and query latency (Tables S5 and S6).

`paper/README.md` maps each of these tables to the script and output file that regenerate it.

## Citation

To be updated on publication.

```bibtex
@article{crve2026,
  title   = {{CRVE}: Low-Risk, {LLM}-Free Query Expansion for Lexical Retrieval from Stored Corpus Contexts},
  author  = {Hoang Van, Dong and Shen, Jialu and Karim, Md Monjurul and Qu, Qiang},
  journal = {to be updated on publication},
  year    = {2026}
}
```

## Requirements

- Python 3.12 (the runs used 3.12.3); the packages are pinned in `requirements.txt`.
- Java 21 (the runs used OpenJDK 21) for Terrier, which PyTerrier downloads on first use (Terrier 5.11, pinned in
  `crve.pipeline.common.pin_terrier`). Not needed for the tables and figures.
- A CUDA GPU only for building context sidecars (`scripts/build_sidecar.py`) and BGE-QE vocabularies; CRVE itself
  encodes queries on the CPU.
- Memory: about 2 GB for the tables and figures; 5 to 8 GB per collection for the pipeline under one million
  documents; up to about 12 GB for the PPMI and profile builds of the five collections over one million documents (if
  built rather than downloaded). The JVM heap is capped at 4 GiB.
- Disk: raw data about 15 GB (plus the BEIR zips while downloading), Terrier indexes of all 25 collections about 27 GB
  (`results/collection_stats.csv`), prebuilt assets 3.8 GB to download and 4.3 GB unpacked.

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Every script is run from the repository root (`python scripts/<x>.py`); the scripts put `src/` on the path themselves.

## Quick start A: tables and figures of the paper (about 1 minute)

```bash
python paper/make_all.py
```

Re-aggregates the stored per-query results under `results/` (no retrieval, no GPU, no Java), recomputes the bootstrap
intervals and Holm tests, cross-checks them against the stored summaries and selected numbers printed in the paper,
writes every table (CSV) and figure (PDF) to `paper/output/` and compares each file byte for byte with `paper/expected/`.
The outputs hold the rows and columns that the paper prints or that a number of its text uses.
`paper/README.md` maps every table, figure and number of the paper to its output file and sources;
`paper/PROVENANCE.md` describes how each stored result was produced.

## Quick start B: CRVE on one collection, end to end

ArguAna as an example (about 5 minutes plus the downloads):

```bash
python scripts/download_data.py --beir arguana        # data/raw/beir/arguana, checked against configs/data_checksums.json
python scripts/build_index.py --corpus arguana        # Terrier index, checked against the frozen statistics
python scripts/fetch_assets.py gate1 sidecars         # pools, lexical profiles, S2 sidecars of all 25 collections (3.4 GB)
python scripts/build_assets.py ppmi --corpus arguana  # PPMI co-occurrence lists
python scripts/run_crve.py --corpus arguana --compare results/feature_sidecar/outcomes_arguana.parquet
```

`run_crve.py` runs Gate 1, Gate 2 and the retrieval of the base and expanded queries over BM25 and DPH, and writes the
candidate lists, the added terms and the per-query metrics (`results/runs/crve/outcomes_arguana.parquet`, arms `S2` and
`base`). `--compare` checks every per-query value against the stored outcomes of the paper and exits with status 1 if
any differs. BRIGHT collections are downloaded with `--bright <domain>` (for example `--bright biology`) and named
`bright_<domain>` everywhere else.

Instead of downloading the assets, they can be built (the profiles then differ slightly from the paper's, see below):

```bash
python scripts/build_assets.py pool --corpus arguana
PYTHONHASHSEED=42 python scripts/build_assets.py profiles --corpus arguana
python scripts/build_sidecar.py --corpus arguana      # GPU
```

## Full reproduction

Collections (code names): `arguana`, `climate_fever`, `dbpedia_entity`, `fever`, `fiqa`, `hotpotqa`, `nfcorpus`, `nq`,
`quora`, `scidocs`, `scifact`, `trec_covid`, `webis_touche2020`, and `bright_<domain>` for `aops`, `biology`,
`earth_science`, `economics`, `leetcode`, `pony`, `psychology`, `robotics`, `stackoverflow`, `sustainable_living`,
`theoremqa_questions`, `theoremqa_theorems`. Over one million documents: `nq`, `dbpedia_entity`, `hotpotqa`, `fever`,
`climate_fever`. Run one collection at a time.

```bash
python scripts/download_data.py                       # all 25 collections, then the checksums (data/README.md)
python scripts/fetch_assets.py                        # all five archives (configs/assets.json)
for c in <collections>; do
  python scripts/build_index.py --corpus $c
  python scripts/build_assets.py ppmi --corpus $c     # only under 1M; the over-1M lists come with the assets
  python scripts/run_crve.py --corpus $c --compare results/feature_sidecar/outcomes_$c.parquet
done
```

Baselines (BM25 and DPH at Terrier's defaults; RM3 and Bo1 as implemented in Terrier, with 10 feedback documents, 10
terms and RM3 λ = 0.5, set explicitly; BGE-QE; LLM-Q2E replayed from the stored keyword cache; tuned RM3
and Bo1 of Supplementary Section S1):

```bash
python scripts/run_baselines.py prf --corpus $c
python scripts/run_qe_baselines.py --corpus $c        # GPU for the BGE-QE query encoding
python scripts/run_baselines.py metrics --corpus $c --compare results/prf_comparison/per_query_$c.parquet
python scripts/run_baselines.py tuned --corpus $c --compare results/prf_oracle/grid_$c.parquet
```

Table 1 and the worked example of Figure 1:

```bash
python scripts/table1.py                              # --raw also counts the judged queries in data/raw
python scripts/figure1_example.py                     # needs the fiqa index, Gate 1 assets and S2 sidecar
```

The run scripts write under `results/runs/` (ignored by git) and only compare with the stored results; the tables and
figures are always made from the stored results by `paper/make_all.py`. The side experiments behind some tables
(sensitivity grids, random-sidecar draws, oracle analyses) are shipped as stored outcomes only (`paper/PROVENANCE.md`).

## What reproduces exactly

With the prebuilt lexical profiles, S2 sidecars and LLM keyword cache, `run_crve.py` and the baseline scripts reproduce
the stored per-query outcomes of the paper exactly (checked by `--compare`). Indexes, pools and PPMI lists are rebuilt
exactly from the raw data (the PPMI file of a rebuild may store the anchors in another order; the lists are identical;
over 1M documents the rebuilt lists, like the prebuilt ones, cover only the anchors of the evaluated queries, and for
these anchors they equal a full computation), and so is the S2 chunk selection; the chunk vectors of a rebuilt sidecar can differ by float16 encoding noise. Three
inputs cannot be rebuilt exactly and are shipped:

- **Lexical profiles.** The profile sampler shares one seeded generator across terms and visits each document's terms
  in Python set order, so a rebuild depends on the string hash seed, which was not recorded for the paper's builds. A
  rebuild with `PYTHONHASHSEED=42` is deterministic but differs from the stored profiles for about one pool term in five
  (SciFact: 80% of the profiles identical; 71 of 250 queries get a different set of added terms; macro nDCG@10 of CRVE
  -0.0007 over BM25 and -0.0041 over DPH).
- **BGE-QE vocabulary.** A rebuild of the vocabulary sidecar can pick another surface form for a few terms (SciFact: 105
  of 12,960 terms, which changes BGE-QE on 1 of 250 queries; NFCorpus rebuilds identically).
- **LLM-Q2E.** The keyword generations (qwen3.5:4b through Ollama) are replayed from the stored cache; a fresh
  generation with the same model and settings differs on about half of the queries. A query missing from the cache
  stops the run.

**Time and disk per stage** (one collection at a time, on the machine used for the paper: 15 GB of memory, one
consumer GPU; the index, CRVE and baseline times were measured for this release):

| Stage | Under 1M documents | Over 1M documents |
|---|---|---|
| Raw data (`download_data.py`) | corpora up to 0.7 GB | corpora of 1.5 to 3.3 GB |
| Terrier index on disk | up to 0.7 GB | 2.4 to 5.8 GB |
| Index (`build_index.py`) | ArguAna 8 s, Biology 8 s | not timed for this release |
| Pool + PPMI + profiles + sidecar (if built) | 14 s to 36 min in all (Table S5) | 62 to 134 min, a lower bound (Table S5) |
| Assets download (`fetch_assets.py`) | 3.8 GB once, for all collections | (same) |
| CRVE run (`run_crve.py`) | ArguAna (1,310 queries) 3.6 min, Biology (53 queries) 20 s | longer (more queries and documents) |
| Baselines (`prf`, `run_qe_baselines.py`, `metrics`) | ArguAna 15 min (BM25, DPH, RM3, Bo1 runs 9.3 min; BGE-QE and LLM-Q2E 5.6 min) | longer |

## Settings

Table 2 of the paper, and where the settings live in the code:

| Setting | Value | Code |
|---|---|---|
| Pool size P, max DF share | 10,000, 0.15 | `evaluation/pool_generators.py` (`is_eligible_term`), `crve/pipeline/assets.py` (`build_pool_v2`) |
| Chunk length, stored chunks n | 100 tokens (85 to 115), 20, lowest IDF density (S2) | `crve/selection/gate2_contracts.py`, `crve/pipeline/sidecar.py` |
| Anchor DF share, M | 0.12, 600 | `crve/selection/gate1_sidecars.py`, `crve/selection/gate1_proposers.py` (`PPMISidecarProposer`) |
| Profiles R, h | 50 windows, 32 terms before / 31 after the occurrence (the centre occurrence of the term removed) | `crve/selection/gate1_sidecars.py` (`build_or_load_sparse_lexical_sidecar`) |
| D, kappa, L | 500, 60, 200 | `crve/pipeline/gate1.py` |
| Encoder, query length | BGE-small-en-v1.5 rev. 5c38ec7c, 128 tokens | `crve/scoring/context_evidence.py` |
| K, w | 5, 0.20 | `crve/pipeline/gate2.py`, `crve/pipeline/retrieval.py` |
| Base retrieval | BM25 / DPH, Terrier defaults, termpipelines=NoOp, 1,000 documents | `crve/pipeline/retrieval.py` |
| RM3 / Bo1 | 10 documents, 10 terms, lambda 0.5 | `evaluation/baselines/pyterrier_harness.py` |

## Repository layout

| Folder | Content |
|---|---|
| `paper/` | Scripts that regenerate the paper's tables and figures from `results/`; expected outputs (`paper/README.md`, `paper/PROVENANCE.md`) |
| `supplementary/` | Supplementary Materials of the paper (PDF and LaTeX source) |
| `results/` | Stored result files of the paper; `results/queries/<c>.txt`: the evaluated queries |
| `src/` | The pipeline: `crve/` (Gate 1, Gate 2, context sidecar, retrieval), `evaluation/` (data loading, pool rules, Terrier baselines) |
| `scripts/` | Command-line steps: data, indexes, assets (build or fetch), CRVE runs, baselines, Table 1 and Figure 1 |
| `configs/` | Gate 1 configuration, frozen Terrier index settings, raw-data checksums, prebuilt-asset list (`assets.json`) |
| `tests/` | Fast unit tests (no Java, no GPU): `python -m pytest tests` |
| `data/` | Raw data, caches and downloaded archives (ignored by git; see `data/README.md`) |

`scripts/mem_guard.sh <min free GiB> <log> -- <command>` runs a command and stops it when free memory runs low.

## License

MIT (see `LICENSE`), copyright the CRVE authors. The collections keep their own licenses (BEIR, BRIGHT); the prebuilt
assets contain passages and terms derived from them.
