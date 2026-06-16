# Vertica → Filtered 30k → Pipeline Timings

Tracking wall-clock for the end-to-end experiment:
1. Filter + sample 30k surviving tables out of the Vertica `main_tokenized` corpus.
2. Run the full pipeline (WP1 → WP2 → WP3 parallel → WP4) on those 30k tables.

All times below are **wall-clock on dama (big-dama-3)** unless noted. The
Vertica DB runs on `big-dama-1`; dama executes the client + the pipeline.

## Reference points (prior runs, for comparison)

| Run | Date | Corpus | Stage / scope | Wall-clock | Notes |
|---|---|---|---|---|---|
| Sequential WP3 | 2026-05-20 | 33,283 WDC candidates (unfiltered) | full `main.py` | **25,701 s (7.14 h)** | dama, single core |
| Parallel WP3 | 2026-06-03 | same 33,283 WDC candidates | `parallel_greedy_partition` only | **6,122 s (102 min)** | dama, 14 workers → ~4.2× speedup |
| ↳ Step 2 only (parallel scoring of 29.66 M edges) | | | | 4,102 s (68 min) | ~6× from 14 cores (long-tail bound) |
| ↳ Step 3 only (serial merge loop, 16,115 merges) | | | | 1,803 s (30 min) | unchanged by parallelism |
| Vertica full column-verdict scan | 2026-06-03 | 8.3 B rows | `SELECT verdict, COUNT(*)` | **1,732 s (29 min)** | summary only, no materialization |
| Paper-conformance WDC | 2026-06-04 | same 33,283 WDC candidates | full `main.py` | **139.75 s (2.33 min)** | dama, 14 workers, `--no_approx`, defaults `--theta_edge=0.85` and `--theta_overlap=3`. ~6.37 M overlap edges (vs 29.66 M), 157,868 positive edges (vs 1.36 M), **24,298 connected components → merge loop in seconds**. Output: 24,323 partitions (89.6% singletons, 2,541 multi-table). |
| ↳ vs sequential baseline | | | | | **184× speedup** |
| ↳ vs parallel-only baseline | | | | | **43.8× speedup** |
| Paper-conformance Vertica 10k | 2026-06-04 | 150,757 candidates from 10k filtered wikitables | full `main.py` | **678.69 s (11.3 min)** | dama, 14 workers, `--no_approx`. 39,940 positive edges (vs prev. 70,352,948, ~1,760× fewer), **145,817 connected components (largest = 182)**. Output: 145,922 partitions (98.7% singletons, 1,903 multi-table). Top mappings: rank→name, rank→score, rank→time. |
| ↳ vs prior 4-day projection on same corpus | | | | | **~510× faster** |

## This experiment

### Phase 1 — Filter + sample 30k tables (Vertica server-side)

| Phase | Start | End | Wall-clock | Notes |
|---|---|---|---|---|
| Connect + submit query | 12:05:42 | 12:05:43 | ~1 s | Direct connection from dama → big-dama-1, no SSH tunnel |
| Vertica scan + filter + sample + stream + JSONL write | 12:05:43 | 13:03:44 | **3,481 s (58.0 min)** | One server-side query; rows stream back and pivot to column-major `relation` per tableid as they arrive. |

**Phase 1 result:** 1,749,143 rows streamed → **30,000 tables** written → 28 MB JSONL (`data/vertica_filtered_30k.jsonl`). Avg ~58 cells per surviving table.

### Phase 2 — Pipeline run on 30k filtered tables

Launched ~13:14 on dama: `main.py --corpus_path data/vertica_filtered_30k.jsonl --output_folder output/vertica_30k/ --threshold 0.3 --theta 0.95 --parallel_workers 14`. PID 1939242.

| Stage | Wall-clock | Notes |
|---|---|---|
| 1 — Load corpus | **2.73 s** | 30,000 tables, 145,040 columns, 584,876 unique values |
| 2 — Build cooccurrence index | **65.79 s** | Fresh build (new corpus, no cached pkl) |
| 3 — Coherence scoring (sequential) | **81.83 s** | main.py uses sequential `score_corpus`, not `parallel_score_corpus` |
| 4 — WP1 filter | **2.50 s** | |
| 5 — WP2 FD filter (sequential) | **27.88 s** | **454,060 candidates** produced — 13.7× the density of WDC (filter keeps cleaner pairs, more pass theta=0.95) |
| 6 — WP3 synthesis (`--parallel_workers 14`) | **FAILED — silent exit** | Died mid-way through `_build_overlap_set` (the sequential pre-parallel step that enumerates O(Σ k²) overlap edges). 454 k candidates → 2.7 M unique pairs, 440 K unique left values → estimated 4-9 billion overlap edges → set doesn't fit in 251 GB. No traceback; almost certainly OOM kill (no sudo access to confirm via `dmesg`). |

**30k attempt aborted.** The current algorithm's `_build_overlap_set` materializes the full overlap-edge set in main-process memory before parallel scoring starts. At 30k filtered tables / 454k candidates, that set exceeds dama's RAM. WP3 step 2 parallelism would have helped if we could get past step 1 (overlap enumeration), but step 1 is the binding constraint here. The deferred Step 3 heap optimization doesn't help either — this is upstream of both.

### Phase 2 retry — 10k filtered tables

#### Extraction

| Phase | Start | End | Wall-clock | Notes |
|---|---|---|---|---|
| Vertica filter + sample + extract (n=10000) | 14:06:35 | 15:03:51 | **3,436 s (57.3 min)** | Same query as before with `LIMIT 10000`. The Vertica scan dominates — basically identical wall-clock to 30k extraction (3,481 s). Streaming back is the only smaller bit. |

**Extraction result:** 588,137 rows → 10,000 tables → 9.6 MB JSONL (`data/vertica_filtered_10k.jsonl`). Avg ~59 cells per surviving table.

#### Pipeline

Launched 15:06 on dama: `main.py --corpus_path data/vertica_filtered_10k.jsonl --output_folder output/vertica_10k/ --threshold 0.3 --theta 0.95 --parallel_workers 14`. PID 1977635.

| Stage | Wall-clock | Notes |
|---|---|---|
| 1 — 5 (load through FD filter) | _fast_ | **150,757 candidates** produced — exactly 1/3 of 30k's 454k (perfect linear scaling) |
| 6 — WP3 synthesis (`--parallel_workers 14`) | _running_ | 150k candidates loaded; pair_index = 919 k, left_index = 159 k |
| 7 — WP4 conflict resolution | _TBD_ | |
| **Total** | _TBD_ | |

## Notes on methodology

- Sampling is **deterministic** via `ORDER BY HASH(tableid) LIMIT 30000` — re-runs return the same 30 k tableids.
- Filter rules at the column level (same as `noise_filter.is_noise`):
  - drop column with <2 unique non-empty values
  - drop column if all non-empty values match `^-?[0-9]+([.,][0-9]+)?$` (pure numeric)
  - drop column if all non-empty values match `^(#|0x)[0-9a-fA-F]+$` (hex)
  - drop column if all non-empty values are placeholders `{-, --, n/a, na, null, none, ?}`
  - drop table with <2 surviving columns
- The full `CREATE TABLE main_tokenized_filtered` materialization was attempted but blocked by Vertica permission (`automap` cannot CREATE in `public`, owned by `olib92`). Switched to a single query that filters + samples + streams in one pass; no materialized artifact.
