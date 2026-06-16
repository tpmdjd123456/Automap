# Auto-Map

Re-implementation of Wang & He, *Synthesizing Mapping Relationships Using
Table Corpus* (SIGMOD 2017). Implements §3 (candidate extraction) and §4
(table synthesis + conflict resolution) end-to-end:

- **§3.1 / WP1** — column filtering by PMI coherence.
- **§3.2 / WP2** — column-pair filtering by approximate functional dependency.
- **§4.2 / WP3** — greedy table synthesis (Algorithm 3).
- **§4.3 / WP4** — conflict resolution (Algorithm 4).

The pipeline runs as 7 stages in `main.py`. Section 5 (evaluation) is out
of scope.

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

### Scaling runs

Paper-faithful exact-match path (fast at any scale):

```bash
python main.py \
    --corpus_path data/vertica_filtered_1500k.jsonl \
    --output_folder output/results_1500k_noapprox/ \
    --threshold 0.3 --theta 0.95 \
    --parallel_workers 8 \
    --max_bucket_size 250 \
    --no_save_index \
    --no_approx
```

Approximate-match path with Jaccard char-2gram (catches fuzzy synonymies
that exact match misses; slower):

```bash
python main.py \
    --corpus_path data/vertica_filtered_1500k.jsonl \
    --output_folder output/results_1500k_jaccard/ \
    --threshold 0.3 --theta 0.95 \
    --parallel_workers 8 \
    --max_bucket_size 250 \
    --no_save_index \
    --string_matcher jaccard
```

### Vertica extract + pipeline chain

`chain_1500k_jaccard.sh` runs the Vertica scan that produces the 1.5M
corpus, then auto-launches the Jaccard pipeline on it. Used for the
1.5M-table run reported in `docs/`:

```bash
./chain_1500k_jaccard.sh
```

The 1.5M Jaccard run wall-clock was 15h 16m on dama (8 workers, 251 GB
RAM) and produced 10,734,449 synthesized mappings of which 112,837 merged
multiple candidates from up to 968 source tables.

### Flags worth knowing

| Flag | Effect |
|---|---|
| `--no_approx` | WP3 uses exact equality (paper-strict §4.1 alternate path). Much faster than approx. |
| `--string_matcher {edit,jaccard,jw}` | Approx matcher when `--no_approx` is not set. `edit` is paper-strict, `jaccard`/`jw` are faster alternatives. |
| `--max_bucket_size N` | Drop inverted-index buckets with >N candidates from WP3 overlap enumeration (caps the Σ k² blowup at scale). |
| `--parallel_workers N` | Parallelize WP3 scoring across N worker processes. |
| `--no_save_index` | Skip pickling the cooccurrence index (avoids 10s-of-GB pickle buffer at 1M+). |
| `--no_skip_noise_values` | Paper-strict cooccurrence index (don't drop pure-numeric / hex / placeholder values). |

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

## Tests

```bash
python -m pytest -v
```

295 tests across WP1 (coherence, NPMI), WP2 (FD filter), WP3 (synthesis,
parallel scoring, connected components, bucket cap), and WP4 (conflict
resolution).

## Repo layout

```
main.py                    7-stage pipeline driver
data_loader.py             JSONL + CSV loaders
cooccurrence_index.py      Interned global value-cooccurrence index (WP1 §3.1)
npmi.py                    PMI / NPMI / coherence math (WP1)
filter.py                  Coherence threshold filter (WP1)
fd_filter.py               Approximate-FD filter (WP2 §3.2)
noise_filter.py            Candidate / value noise predicates
synthesis.py               Greedy partition + scoring (WP3 §4.2)
parallel_pipeline.py       Parallel WP3 scoring + greedy_partition
conflict_resolution.py     WP4 §4.3
heartbeat.py               CPU/RAM heartbeat for long stages
jaccard_similarity.py      Jaccard char-2gram matcher
jaro_winkler.py            Jaro-Winkler matcher
string_matcher.py          Edit-distance + helpers
extract_filtered_sample.py Vertica server-side filter + sample
run_wp4.py                 Stand-alone WP4 runner
chain_1500k_jaccard.sh     Extract -> pipeline chain for 1.5M runs
tests/                     295-test pytest suite
data/sample.json           Bundled small sample corpus
output/                    Gitignored
```

## Documentation

- `docs/` — run timings (`*-vertica-*-timings.md`) and scaling notes.
- `claude/` — design specs, walkthrough, runbook, deviations log.
- `papers/automap.pdf` — source paper.
