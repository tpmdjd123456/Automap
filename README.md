# Auto-Map

Re-implementation of Wang & He, *Synthesizing Mapping Relationships Using
Table Corpus* (SIGMOD 2017). Implements §3 (candidate extraction) and §4
(table synthesis + conflict resolution) end-to-end:

- **§3.1 / WP1** — column filtering by PMI coherence.
- **§3.2 / WP2** — column-pair filtering by approximate functional dependency.
- **§4.2 / WP3** — greedy table synthesis (Algorithm 3).
- **§4.3 / WP4** — conflict resolution (Algorithm 4).

The pipeline runs as 7 stages in `main.py` (load corpus, co-occurrence
index, coherence scores, column filtering, FD filtering, synthesis,
conflict resolution).

## Final result

**86.06% covered recall (25,933 / 30,134)** on a 1.5M-table sample of the
WDC 2015 English relational web-table corpus, evaluated against the
web-table benchmark (95,088 gold pairs). Run completed 2026-07-06 on dama
in 10.5 h (8 workers, peak leaving 71 GB of 251 GB RAM available).

The sample was built with the **covered-sample method**: the pipeline can
only ever seed a mapping from a row co-occurrence, so the corpus sample
includes every *cover table* (a RELATION table in which a benchmark gold
pair co-occurs in a row) plus random filler to 1.5M tables. Only 31.7% of
the benchmark pairs co-occur anywhere in our 2M-table archive at all —
that coverage, not the algorithm, bounds global recall. Of the pairs that
are discoverable, the pipeline finds 86%. This is flat vs. the
covered-100k baseline (86.2%): recall is insensitive to both the scoring
pair-cap and corpus scale. The paper's 0.88 recall on ~100M tables is
consistent with this — global recall is bounded by corpus coverage, not
by the method.

| Metric | covered-100k (baseline) | covered-1.5M (final) |
|---|---|---|
| Covered recall (vs 30,134 reachable pairs) | 86.2% | **86.06%** |
| Global recall (vs all 95,088 gold pairs) | 27.3% | 27.3% |
| Scoped precision (within benchmark value universe) | 16% | 8.5% |
| Wall clock | 4.6 h | 10.5 h |

Full manifest (exact invocation, per-stage timings, resource traces,
output stats, interpretation): **`samples/wdc_1500k_final/RESULTS.md`**.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Small smoke test on the bundled sample corpus (~2,700 tables, ~9 s):

```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/sample/ \
               --threshold 0.3 --theta 0.95
```

### Final-run configuration (covered-1.5M, capped)

The exact configuration of the final run (`run_covered_1500k_capped_driver.sh`
is the exact driver, including resource pre-flight and memory logging):

```bash
export AUTOMAP_SCORE_PAIR_CAP=150
python -u main.py \
    --corpus_path data/wdc_covered_1500k.jsonl \
    --output_folder output/wdc_covered_1500k_capped/ \
    --threshold 0.3 --theta 0.95 \
    --parallel_workers 8 \
    --max_bucket_size 250 \
    --no_save_index \
    --string_matcher jaccard
```

All other parameters at their defaults: `--tau -0.2`, `--theta_overlap 3`,
`--theta_edge 0.85`, `--min_rows 3`, `--table_types RELATION`.

`AUTOMAP_SCORE_PAIR_CAP=150` caps the number of value pairs per candidate
used for edge *scoring* only (bounding each edge computation to O(cap²)
instead of O(|B|·|B'|)); full pair lists still flow to the output, so
recall is unaffected (86.2% at 100k, identical to uncapped). Without the
cap, the 1.5M run peaked at ~233/251 GB RAM in Stage 2 and projected
~4.5 days for edge scoring; with it, 10.5 h total with 71 GB headroom.

### Building the covered corpus and benchmark subset

```bash
# 1. Which gold pairs co-occur in a row anywhere in the archive?
#    -> data/benchmark-covered.txt (30,134 pairs = 31.7% of benchmark)
python extract_covered_pairs.py data/benchmark-web.txt <corpus.jsonl> data/benchmark-covered.txt

# 2. Cover tables for those pairs (from the WDC archives)
python build_benchmark_cover.py

# 3. Corpus sample = all cover tables + random filler to N tables
#    (streaming variant for the 1.5M build; 6.3 GB output)
python build_covered_sample_stream.py

# WDC raw-archive -> JSONL conversion (first N tables of an archive):
python make_wdc_jsonl.py
```

### Evaluation

```bash
python eval_benchmark.py data/benchmark-web.txt <resolved_mappings.jsonl> \
                         <corpus.jsonl> data/benchmark-covered.txt
```

Reports global precision/recall/F1 vs. all gold pairs, *achievable* recall
(both values present in the corpus), *covered* recall (pair co-occurs in a
row — the true recall ceiling), and scoped precision within the benchmark
value universe. The 4th argument (covered subset) is optional.

### Earlier scaling runs (Vertica corpus)

Before the WDC work, the pipeline was scaled on a 1.5M-table Vertica
enterprise corpus. `chain_1500k_jaccard.sh` runs the Vertica scan that
produces that corpus, then auto-launches the Jaccard pipeline on it
(15h 16m wall clock, 10,734,449 synthesized mappings; see `docs/`).
`run_wdc_calibration.sh` measures WDC scaling at 10k/50k/250k.

### Parameters

| Flag | Default | Effect |
|---|---|---|
| `--threshold` | 0.3 | WP1 PMI-coherence column-filter threshold. |
| `--theta` | 0.95 | WP2 approximate-FD threshold. |
| `--min_rows` | 3 | WP2 minimum rows for a column pair to be considered. |
| `--tau` | -0.2 | WP3 synthesis score threshold (Algorithm 3). |
| `--theta_overlap` | 3 | WP3 minimum value overlap between candidates. |
| `--theta_edge` | 0.85 | WP3 edge-consistency threshold. |
| `--no_approx` | off | WP3 uses exact equality (paper-strict §4.1 alternate path). Much faster than approx. |
| `--string_matcher {edit,jaccard,jw}` | edit | Approx matcher when `--no_approx` is not set. `edit` is paper-strict, `jaccard`/`jw` are faster alternatives. |
| `--max_bucket_size N` | 0 (off) | Drop inverted-index buckets with >N candidates from WP3 overlap enumeration (caps the Σ k² blowup at scale). |
| `--parallel_workers N` | 1 | Parallelize WP3 scoring across N worker processes. |
| `--no_save_index` | off | Skip pickling the cooccurrence index (avoids 10s-of-GB pickle buffer at 1M+). |
| `--no_skip_noise_values` | off | Paper-strict cooccurrence index (don't drop pure-numeric / hex / placeholder values). |
| `--table_types` | RELATION | WDC table types to load. |
| `--index_path` / `--rebuild_index` | — | Reuse / force-rebuild a pickled cooccurrence index. |

| Env var | Default | Effect |
|---|---|---|
| `AUTOMAP_SCORE_PAIR_CAP` | 0 (off) | Cap value pairs per candidate in WP3 edge *scoring* (read in `parallel_pipeline.py`). Bounds each edge to O(cap²); output pair lists are unaffected, so recall is preserved. 150 used for the final run. |

### Reproducing just WP4

If a pipeline run was killed after WP3 saved `synthesized_mappings.jsonl`
but before WP4 finished, recover the WP4 output without redoing WP1-WP3:

```bash
python run_wp4.py \
    --synthesized output/<folder>/synthesized_mappings.jsonl \
    --resolved   output/<folder>/resolved_mappings.jsonl
```

## Outputs

All artifacts land in `--output_folder`:

| Path | What |
|---|---|
| `filtered_corpus.jsonl` | WP1 output — corpus with low-coherence columns removed. |
| `coherence_distribution.png` | Histogram of column coherence with the threshold line. |
| `threshold_sweep.txt` | Kept-vs-removed at thresholds 0.1, 0.2, 0.3, 0.4, 0.5. |
| `cooccurrence_index_{skipnoise,full}.pkl` | Pickled index (omitted under `--no_save_index`). |
| `candidates.jsonl` | WP2 output — one column pair per line. Input to WP3. |
| `computed_edge_scores.jsonl` | WP3 edge weights (for inspection). |
| `synthesized_mappings.jsonl` | WP3 output — synthesized partitions of candidates. |
| `resolved_mappings.jsonl` | WP4 output — partitions after conflict resolution. |

## Results in this repo vs. on dama

- **`samples/wdc_1500k_final/`** — the final run's record: `RESULTS.md`
  manifest (exact invocation, timings, resources), 1,000-record uniform
  random samples of `resolved_mappings.jsonl` and
  `synthesized_mappings.jsonl`, the full benchmark eval, threshold sweep,
  and coherence distribution.
- **Full outputs** (not in git): `dama:/home/automap/Automap/results/` —
  final `resolved_mappings.jsonl` (3.5 GB, 4,144,953 records), full
  pipeline log, memory traces, driver log. Intermediates in
  `dama:/home/automap/Automap/output/wdc_covered_1500k_capped/`.

## Tests

```bash
python -m pytest -v
```

295 tests across WP1 (coherence, NPMI), WP2 (FD filter), WP3 (synthesis,
parallel scoring, connected components, bucket cap), and WP4 (conflict
resolution).

## Repo layout

```
main.py                        7-stage pipeline driver
data_loader.py                 JSONL + CSV loaders
cooccurrence_index.py          Interned global value-cooccurrence index (WP1 §3.1)
npmi.py                        PMI / NPMI / coherence math (WP1)
filter.py                      Coherence threshold filter (WP1)
fd_filter.py                   Approximate-FD filter (WP2 §3.2)
noise_filter.py                Candidate / value noise predicates
synthesis.py                   Greedy partition + scoring (WP3 §4.2)
parallel_pipeline.py           Parallel WP3 scoring + greedy_partition (+ pair cap)
conflict_resolution.py         WP4 §4.3
heartbeat.py                   CPU/RAM heartbeat for long stages
jaccard_similarity.py          Jaccard char-2gram matcher
jaro_winkler.py                Jaro-Winkler matcher
string_matcher.py              Edit-distance + helpers
eval_benchmark.py              Benchmark eval: global / achievable / covered recall
extract_covered_pairs.py       Which gold pairs co-occur in a corpus row
build_benchmark_cover.py       Cover tables for the covered gold pairs
build_covered_sample.py        Covered corpus sample (in-memory)
build_covered_sample_stream.py Covered corpus sample (streaming, used for 1.5M)
make_wdc_jsonl.py              WDC archive -> JSONL conversion
show_mappings.py               Inspect mapping outputs
dedup_mappings.py              Dedup mapping outputs
extract_filtered_sample.py     Vertica server-side filter + sample
run_wp4.py                     Stand-alone WP4 runner
run_covered_*_driver.sh        dama drivers (pre-flight checks + memory logging)
run_wdc_calibration.sh         WDC scaling measurement (10k/50k/250k)
chain_1500k_jaccard.sh         Vertica extract -> pipeline chain for 1.5M runs
tests/                         295-test pytest suite
data/sample.json               Bundled small sample corpus
data/benchmark-covered.txt     Covered benchmark subset (30,134 pairs)
samples/                       Committed run samples (see above)
output/                        Gitignored
```

## Documentation

- `docs/` — run timings (`*-vertica-*-timings.md`) and scaling notes.
- `claude/` — design specs, walkthrough, runbook, deviations log.
- `samples/wdc_1500k_final/RESULTS.md` — final-run manifest.
