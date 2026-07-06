# Automap on WDC Web Tables — Final Results (covered-1.5M capped run)

**Date:** pipeline run 2026-07-05 17:16 → 2026-07-06 03:50 (exit code 0); benchmark eval 2026-07-06.
**This folder is the canonical record of the project's final result on dama.**

## What this is

Automap (value-mapping discovery pipeline) run on a 1.5M-table sample of the WDC
web-table corpus, evaluated against the web-table benchmark. The sample was built
with the **covered-sample method**: it includes every *cover table* — a RELATION
table in which a benchmark gold pair co-occurs in a row — because the pipeline can
only ever seed a mapping from a row co-occurrence. Earlier low recall (~5%) on
natural samples was a sampling artifact, not an algorithm failure.

**Headline result: 86.06% covered recall (25,933 / 30,134)** — the pipeline finds
86% of all benchmark pairs that are discoverable in this corpus at all. This is
flat vs. the covered-100k baseline (86.2%), i.e. recall is insensitive to both
the scoring pair-cap and to corpus scale; the remaining ~14% is lost to the
algorithm/filters, not to corpus size. The paper's 0.88 on ~100M tables is
consistent with this: headline (global) recall is bounded by corpus coverage
(31.7% ceiling on our 2M-table archive), not by the method.

## Inputs

| Item | Value |
|---|---|
| Corpus | `data/wdc_covered_1500k.jsonl` (6.3 GB; 1,492,094 tables loaded, 6,503,066 columns, 14,550,457 unique values) |
| Corpus construction | `build_covered_sample_stream.py`: all 30,134-pair cover tables from `data/wdc_archives/` (WDC 2015 RELATION, archives 00+01, ~2M tables) + random filler to 1.5M |
| Benchmark | `data/benchmark-web.txt` — 95,088 distinct unordered gold pairs |
| Covered subset | `data/benchmark-covered.txt` — 30,134 gold pairs (31.7%) that co-occur in a row somewhere in the 2M archive (built by `extract_covered_pairs.py`); re-verified = 30134 at launch |

## Exact invocation

```
export AUTOMAP_SCORE_PAIR_CAP=150
.venv/bin/python -u main.py \
    --corpus_path data/wdc_covered_1500k.jsonl \
    --output_folder output/wdc_covered_1500k_capped/ \
    --threshold 0.3 --theta 0.95 --parallel_workers 8 \
    --max_bucket_size 250 --no_save_index --string_matcher jaccard
```

Launched by `driver_script.sh` (copy in this folder) under `setsid`, bare —
**no memory watchdog kill** (explicitly requested; a passive memory logger ran
instead, see `memlog_pipeline.log`). Attempt #1 without the pair cap had peaked
at ~233/251 GB in Stage 2 and was projected at ~4.5 days for edge scoring.

### Config summary

| Parameter | Value | Note |
|---|---|---|
| `AUTOMAP_SCORE_PAIR_CAP` | 150 | env var, read in `parallel_pipeline.py` `_score_edge_worker`; caps pairs-per-candidate for edge *scoring* only (bounds each edge to O(cap²)); full pairs still flow to output, so recall is preserved |
| `--threshold` | 0.3 | PMI coherence column filter |
| `--theta` | 0.95 | FD filtering (min_rows=3) |
| `--parallel_workers` | 8 | |
| `--max_bucket_size` | 250 | |
| `--string_matcher` | jaccard | |
| synthesis tau | -0.2 | pipeline default |
| Code version | commit `0103e1f` + rsync-synced local modifications to `main.py`, `parallel_pipeline.py`, `synthesis.py` (pair-cap support etc.) | code synced via rsync, not commits |

## Timings (total 37,944 s = 10.54 h)

| Stage | Time | Notes |
|---|---|---|
| 1. Load corpus | 240 s | 1,492,094 tables |
| 2. Co-occurrence index | 2,305 s | |
| 3. Coherence scores | 4,695 s | |
| 4. Column filtering (PMI) | 86 s | threshold sweep in `threshold_sweep.txt`; 61.5% kept at 0.3 |
| 5. FD filtering | 930 s | 958,924 source tables represented |
| 6. Table synthesis | 29,077 s | dominant stage (77% of wall clock) |
| 7. Conflict resolution | 577 s | |

Coverage pre-check before launch (`extract_covered_pairs.py`): ~6 min.
Benchmark eval (`eval_benchmark.py`, single-threaded): ~35 min, peak RSS ~23 GB.

## Resources (shared box, 251 GB RAM)

- Pre-flight: 61 GB disk free, 246 GB MemAvailable, load 2.04.
- During run: minimum MemAvailable **71 GB** (vs. ~18 GB headroom on uncapped
  attempt #1) — the pair cap fixed memory as well as runtime. 30 GB disk free at end.
- Full 30-second-interval memory trace: `memlog_pipeline.log` (1,265 samples).

## Pipeline output stats

- Synthesized mappings: **4,144,953** across 3,853,599 converged components;
  95.8% singleton partitions, 174,584 multi-table.
- Pairs per mapping: min 2, mean 14.4, max 10,817 (largest blobs are known junk
  shapes: store-locator "N in-stock at <city>" clusters, currency-rate tables).
- Conflict resolution: 24,498 mappings had conflicts; 429,427 pairs removed.
- Final output: `resolved_mappings.jsonl` (3.5 GB, 4,144,953 records; hardlinked
  from `output/wdc_covered_1500k_capped/`).

## Benchmark evaluation (`eval_1500k.txt`)

```
eval_benchmark.py data/benchmark-web.txt results/resolved_mappings.jsonl \
                  data/wdc_covered_1500k.jsonl data/benchmark-covered.txt
```

| Metric | covered-100k (baseline) | **covered-1.5M (this run)** |
|---|---|---|
| Covered recall (vs 30,134 reachable pairs) | 86.2% (25,969) | **86.06% (25,933)** |
| Global recall (vs all 95,088 gold pairs) | 27.3% | 27.3% |
| Achievable recall denominator (both values in subset) | — | 51,863 pairs → 50.0% |
| Predicted distinct pairs | 5.5 M | 25.9 M |
| Scoped precision (within benchmark value universe) | 16% | 8.5% |
| Scoped F1 | — | 0.145 |
| Wall clock | 4.6 h | 10.5 h |

Interpretation: 15× more corpus recovered the *same* covered pairs (36 fewer,
noise-level) while growing predicted pairs ~5×, halving scoped precision.
Scale adds noise, not recall.

## Files in this folder

| File | What it is |
|---|---|
| `RESULTS.md` | this manifest |
| `resolved_mappings.jsonl` | **final pipeline output** (3.5 GB, hardlink) |
| `eval_1500k.txt` | benchmark eval of this run (headline numbers) |
| `eval_100k_baseline.txt` | covered-100k eval, for comparison |
| `driver_script.sh` | exact driver that launched the run (config + watchdog logic) |
| `driver.log` | driver timeline: pre-flight, launch, exit code 0 |
| `pipeline_run.log` | full pipeline stdout/stderr (79 MB, hardlink) |
| `memlog_pipeline.log` | 30 s MemAvailable/disk trace during the run |
| `memlog_eval.log` | 5 min RSS/CPU trace during the eval |
| `threshold_sweep.txt` | Stage 4 PMI threshold sweep |
| `coherence_distribution.png` | Stage 3 coherence score distribution |

Intermediates (candidates 17 GB, filtered corpus 5.9 GB, edge scores 3 GB,
pre-resolution synthesized mappings 3.7 GB) remain in
`output/wdc_covered_1500k_capped/` and are safe to delete if disk is needed —
everything needed to interpret or re-run the result is in this folder.
