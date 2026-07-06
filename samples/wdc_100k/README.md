# WDC Web-Table Corpus — 100k run

Pipeline run on a 100k-table subset of the WDC English **relational** web-table
corpus (Web Data Commons). The corpus archives (`00.tar.gz`, `01.tar.gz`, ~2.0M
tables total = 2 of the full 51 archives) are too large for git and live on the
`dama` box under `data/wdc_archives/`; convert them to JSONL with
`make_wdc_jsonl.py`.

## Config (matches the 1.5M Vertica run)

```
main.py --threshold 0.3 --theta 0.95 \
        --parallel_workers 8 \
        --max_bucket_size 250 \      # cap that bounds the WP3 O(k^2) blowup
        --string_matcher jaccard \
        --no_save_index
```

The `--max_bucket_size 250` cap is the key heuristic — without it WP3 edge
weighting crawls (~500 pairs/s vs ~10k pairs/s capped).

## Pipeline counts

| Stage | Metric | Count |
|-------|--------|------:|
| Input | Table IDs | 100,000 |
| Loaded | Tables (RELATION + non-degenerate cols) | 99,411 |
|  | Columns / unique values | 431,157 / 1,595,114 |
| Candidates (WP1/WP2) | Column-pair candidates | 307,446 |
| Synthesis (WP3) | Synthesized mappings | 262,308 |
| Conflict res (WP4) | Mappings with conflicts | 1,749 |
|  | Value-pairs removed | 31,270 |
|  | Mappings dropped | 0 |
| **Final** | Resolved mappings | **262,308** |

Of the 262,308 final mappings: **143,223 distinct** (after removing identical +
symmetric duplicates), **9,080 cross-table** (>=2 source tables), 253,228
singletons.

## Timing (8 workers, dama)

Total **40.2 min** (2,409s). WP3 = 1,838s (76%). Index 156s, coherence 302s,
FD filter 62s. Peak RAM ~12 GB.

## Benchmark evaluation

Ground truth: `benchmark-web.txt` — 80 binary relations, 95,088 distinct gold
pairs. See `benchmark_eval.txt` for raw output.

| Metric | Value | Note |
|--------|------:|------|
| Global precision | 0.0022 | misleading — benchmark covers only 80 relations |
| Global recall | 0.0515 | capped: only 20.4% of gold pairs are in this subset |
| Achievable recall | **0.2522** | of gold pairs whose values exist in the subset |
| Scoped precision | **0.1550** | predicted pairs within the benchmark value universe |
| Scoped F1 | **0.1920** | |

Global P/R understate quality: (a) the benchmark is a partial gold standard
(80 relations) while the pipeline extracts correspondences for all columns, and
(b) this is only ~5% of the 2M-table data, so most gold pairs physically cannot
be found here. Recall should rise with larger subsets (500k/750k planned).

## Files

- `resolved_mappings.sample.jsonl` — first 3,000 final mappings
- `synthesized_mappings.sample.jsonl` — first 3,000 pre-conflict-resolution mappings
- `benchmark_eval.txt` — full P/R/F1 output
- `coherence_distribution.png` — coherence-score histogram

## Scripts (repo root)

- `make_wdc_jsonl.py` — stream first N tables from a WDC archive into JSONL
- `run_wdc_calibration.sh` — measure pipeline scaling at 10k/50k/250k
- `eval_benchmark.py` — precision/recall/F1 vs a ground-truth benchmark
- `show_mappings.py` — print readable sample mappings
- `dedup_mappings.py` — count distinct vs duplicate/symmetric mappings
