# Data

`data/raw/` and `data/cache/` are ignored by git. Regenerating the paper's tables and figures (`python paper/make_all.py`)
needs nothing from this folder: it reads only the stored results under `results/` (document counts and corpus sizes are
frozen in `results/collection_stats.csv`).

## Raw data (`data/raw/`)

```bash
python scripts/download_data.py                  # all 25 collections, then the checksums
python scripts/download_data.py --verify-only    # only the checksums
```

| Collections | Source | Files read |
|---|---|---|
| 13 BEIR: ArguAna, Climate-FEVER, DBPedia-Entity, FEVER, FiQA, HotpotQA, NFCorpus, NQ, Quora, SCIDOCS, SciFact, TREC-COVID, Touché-2020 | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip` | `beir/<name>/corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` |
| 12 BRIGHT: AoPS, Biology, Earth Science, Economics, LeetCode, Pony, Psychology, Robotics, Stack Overflow, Sustainable Living, TheoremQA-Questions, TheoremQA-Theorems | Hugging Face `xlangai/BRIGHT`, revision `3066d29c9651a576c8aba4832d249807b181ecae` | `bright/documents/<domain>-00000-of-00001.parquet`, `bright/examples/<domain>-00000-of-00001.parquet` |

`configs/data_checksums.json` holds the SHA-256 of these 63 files as used for the paper. The stored Gate 1 sidecars and
context sidecars record the corpus hash and are only loaded against identical corpora.

Collection names in the code: the BEIR names with `_` instead of `-` (`climate_fever`, `dbpedia_entity`, `trec_covid`,
`webis_touche2020`) and `bright_<domain>`. Query ids: `q_beir_<collection>_<id>` and `q_bright_<domain>_<id>`.

## Caches (`data/cache/`)

| Path | Built by | Prebuilt archive (`scripts/fetch_assets.py <key>`) |
|---|---|---|
| `terrier_indices/<c>_default/` | `scripts/build_index.py` | none (rebuilt exactly) |
| `gate1/<c>_canonical_pool.json` | `scripts/build_assets.py pool` | `gate1` (`crve_gate1_assets.tar.gz`) |
| `gate1/<c>_bounded_ppmi.parquet` | `scripts/build_assets.py ppmi` | `ppmi` (`crve_ppmi_over1m.tar.gz`): the five collections over 1M documents (hours to rebuild) |
| `gate1/<c>_lexical_profiles.parquet`, `gate1/<c>_lexical_profiles_idx/` | `PYTHONHASHSEED=42 scripts/build_assets.py profiles` | `gate1`: **needed to reproduce the paper's numbers exactly** |
| `sidecars/S2/<c>/` | `scripts/build_sidecar.py` (GPU) | `sidecars` (`crve_s2_sidecars.tar.gz`; saves the encoding) |
| `qe_llm/raw/<hash>.json` | LLM-Q2E generations (qwen3.5:4b through Ollama) | `llm` (`crve_qe_llm_cache.tar.gz`): **replayed; not regenerable** |
| `qe_vocab/<c>/<id>/` | built on first use by `scripts/run_qe_baselines.py` (GPU) | `vocab` (`crve_qe_vocab.tar.gz`): a rebuild can pick other surface forms for a few terms |
| `runs/<c>_<pipeline>.parquet` | `scripts/run_baselines.py prf`, `scripts/run_qe_baselines.py` | none |

`python scripts/fetch_assets.py` downloads the five archives from Google Drive (3.8 GB; file ids, sizes and SHA-256 in
`configs/assets.json`), checks their SHA-256 and unpacks them into the repository root (4.3 GB under `data/cache/`); each
archive covers all 25 collections. Archives downloaded by hand are checked and unpacked where they are with `--from-dir <folder>`. The Gate 1
sidecar loader checks the pool hash, the document count, the corpus hash, the analyser version and the reservoir seed of
every file it reads, and the context sidecar loader checks the SHA-256 of its three files.

In the archives, the first line of each profile index's `data.properties` (a comment holding the build path, ignored by
Terrier) was removed, and the free-text `note` in the metadata of the five over-1M PPMI files was reworded (it named an
internal document); the table data are unchanged. The `pool_path` field of the context sidecar manifests records the
path of the pool at build time and is not read by the pipeline.
