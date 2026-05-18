# Auto-Map — Section 3: Candidate Table Extraction

Re-implementation of **Section 3** of Wang & He, *Synthesizing Mapping
Relationships Using Table Corpus* (SIGMOD 2017): the candidate-extraction
preprocessing stage that produces mapping-pair candidates ready for table
synthesis.

Section 3 has two sub-steps and both are implemented here:

- **§3.1 Column filtering by PMI** (WP1) — drop incoherent columns whose
  values don't semantically belong together.
- **§3.2 Column-pair filtering by FD** (WP2) — for every pair of surviving
  columns in a table, keep only those that satisfy approximate functional
  dependency (`X →_θ Y`, θ ≥ 0.95).

Section 4.2 (Table Synthesis) is implemented in `synthesis.py`.
Section 4.3 (Conflict Resolution) is implemented in `conflict_resolution.py`.
Section 5 (Evaluation) is out of scope for this repo.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

One command runs the full Section 3 pipeline:

```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/ \
               --threshold 0.3 \
               --theta 0.95 \
               --index_path output/cooccurrence_index.pkl
```

Five stages: load → build co-occurrence index → score columns → WP1 filter
→ WP2 FD filter. On the sample corpus (~2,700 tables) this runs in ~9
seconds end-to-end and produces ~33,000 candidate column pairs.

CSV folder input also works (`--corpus_path path/to/csv_folder/`).
Re-runs reuse the cached index from `--index_path` unless you pass
`--rebuild_index`.

## Outputs

All artifacts land in `--output_folder`:

| Path | What |
|---|---|
| `filtered_corpus.jsonl` | WP1 output — corpus with low-coherence columns removed. Mirrors input schema; adds `coherence_scores` and `rejected_column_indices`. |
| `coherence_distribution.png` | Histogram of column coherence scores with a red dashed line at the threshold. |
| `threshold_sweep.txt` | Kept-vs-removed column counts at thresholds {0.1, 0.2, 0.3, 0.4, 0.5}. |
| `cooccurrence_index.pkl` | Pickled `(cooccurrence, value_count, total_columns)`. Cached for re-runs. |
| `candidates.jsonl` | **WP2 output** — one ordered column pair per line, each with deduplicated `(left, right)` value pairs, `theta`, `row_count`, `covered_rows`, source-table indices, and pass-through metadata. This is the input to WP3. |

## How it works

1. **Load** the corpus (JSONL or CSV folder). For JSONL, only `RELATION`
   tables are kept and the header row is stripped per `hasHeader` /
   `headerRowIndex` metadata.
2. **Index** every distinct value and every distinct value pair across all
   columns of the corpus (one column = one observation; pairs are stored
   sorted so `(a,b)` and `(b,a)` collide).
3. **Score** each column by mean NPMI over all unordered pairs of its
   distinct values.
4. **WP1 — Filter columns** below the coherence threshold τ. Rebuild
   surviving tables and emit `filtered_corpus.jsonl`.
5. **WP2 — Filter column pairs** by approximate FD. For every ordered
   pair `(C_i, C_j)` from each surviving table, compute the witness-subset
   θ score. Pairs with θ ≥ 0.95 become candidates and are emitted to
   `candidates.jsonl`.

## Tests

```bash
python -m pytest -v
```

65 tests covering: value normalization, JSONL/CSV loaders, co-occurrence
index, PMI/NPMI math, coherence filtering, approximate-FD filtering,
candidate schema, and an end-to-end smoke test that runs `main.py` via
subprocess on a synthetic corpus.

## Repo layout

```
main.py                        Section 3 pipeline driver (5 stages)
data_loader.py                 JSONL + CSV loaders, value normalization
cooccurrence_index.py          Global co-occurrence index over the corpus
npmi.py                        PMI / NPMI / coherence math (WP1)
filter.py                      Coherence threshold filter + reporting (WP1)
fd_filter.py                   Approximate-FD filter + candidate emission (WP2)
tests/                         pytest suite (test_wp1.py, test_wp2.py)
conftest.py                    Shared synthetic fixtures
data/sample.json               WDC web-table sample (~30 MB)
conflict_resolution.py         Conflict resolution pipeline (WP4)
output/                        Pipeline outputs (gitignored)
```

## Documentation

- **`claude/USAGE.md`** — runbook with all CLI flags, output schemas,
  inspection snippets, and θ/τ tuning guidance.
- **`claude/WALKTHROUGH.md`** — code walkthrough explaining the math, the
  module boundaries, the synthetic test corpus design, and key Q&A.
- **`claude/deviations.md`** — log of every place this implementation
  departs from the original prompt or makes a non-obvious design choice,
  with rationale.
- **`docs/superpowers/specs/`** — design specs for WP1 and WP2.
- **`docs/superpowers/plans/`** — implementation plans (TDD task lists).
- **`papers/automap.pdf`** — source paper (Section 3 is what's
  re-implemented here).
