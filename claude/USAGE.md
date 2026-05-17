# WP1 Usage Guide

How to run the PMI coherence-filtering pipeline and how to interpret what comes out.

---

## 1. Setup

WP1 needs Python 3.8+ and four packages: `pandas`, `numpy`, `matplotlib`, `pytest`.

```bash
# From the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have a system-level Python with these packages, you can skip the venv. Tests assume the venv exists at `.venv/`; CI or other Pythons just need the four packages on `PATH`.

---

## 2. Running the pipeline

`main.py` runs the entire Section 3 pipeline (WP1 PMI filtering + WP2 FD filtering) in a single command:

```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/ \
               --threshold 0.3 \
               --theta 0.95 \
               --index_path output/cooccurrence_index.pkl
```

You'll see five stage banners:

```
[Stage 1/5] Loading corpus from data/sample.json...
  Loaded 2716 tables, 11956 columns, 66871 unique values
  Avg columns per table: 4.40
  Time: 0.27s

[Stage 2/5] Building co-occurrence index...
  Saved index to output/cooccurrence_index.pkl
  Processed 11956 columns
  Found 4874328 unique value pairs
  Top co-occurring pairs:
    (..., ...): N
  ...
  NPMI sanity: top pair (...) count=N npmi=1.000
  NPMI sanity: symmetric on first 100 pairs: True
  NPMI sanity: in-range on first 200 pairs: True
  Time: 3.50s

[Stage 3/5] Computing coherence scores...
  Scored 11956 columns
  Average coherence score: 0.778
  Highest: [...] -> 1.000
  Lowest:  [...] -> -0.184
  Time: 4.61s

[Stage 4/5] Filtering columns (PMI coherence)...
  Total columns before filtering: 11956
  Total columns after filtering: 11609
  Removed: 347 (2.9%)
  Saved filtered corpus to output/filtered_corpus.jsonl
  Saved histogram to output/coherence_distribution.png
  Saved threshold sweep to output/threshold_sweep.txt
  Time: 0.46s

[Stage 5/5] FD filtering (theta=0.95, min_rows=3)...
  Candidates: 33283
  Source tables represented: 2128
  Theta: mean=0.999, min=0.950, max=1.000
  Top 5 by theta: ...
  Saved 33283 candidates to output/candidates.jsonl
  Time: 1.53s

Section 3 Complete! Candidates ready for WP3 (table synthesis).
Total time: 8.54s
```

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--corpus_path` | required | Path to a JSONL file (web-table corpus) **or** a folder of CSVs. Auto-dispatches on `os.path.isdir/isfile`. |
| `--output_folder` | required | Where all output artifacts land. Created if missing. |
| `--threshold` | `0.3` | WP1 coherence cutoff. Columns with score `>= threshold` are kept; `< threshold` removed. |
| `--theta` | `0.95` | WP2 approximate-FD threshold. Column pairs with θ `< --theta` are rejected. |
| `--min_rows` | `3` | WP2 minimum non-empty rows for a column pair to be evaluated. Below this, FD has no meaning. |
| `--index_path` | `<output_folder>/cooccurrence_index.pkl` | Where to save (or load from) the pickled co-occurrence index. |
| `--rebuild_index` | off | Force rebuild even if a cached index exists at `--index_path`. Use this whenever the corpus changes. |
| `--table_types` | `RELATION` | Comma-separated `tableType` values to load from JSONL. The default mirrors the paper's assumption that input tables are relational. Pass e.g. `RELATION,ENTITY` to broaden. |

### Re-runs are fast

The first run builds the co-occurrence index and pickles it. Subsequent runs load the cached index in milliseconds, so iterating on the threshold (or anything downstream of indexing) is cheap:

```bash
# First run (slow): builds and caches the index
python main.py --corpus_path data/sample.json --output_folder output/ --threshold 0.3

# Second run (fast): same index, different threshold
python main.py --corpus_path data/sample.json --output_folder output/ --threshold 0.5
```

If you change the corpus, pass `--rebuild_index` or delete the pickle.

---

## 3. Input formats

### JSONL (primary)

One JSON object per line, in the WDC web-table format. The loader uses these fields:

```jsonc
{
  "relation": [["a","b","c"], ["x","y","z"]],   // column-major: each inner list is one column
  "tableType": "RELATION",                       // or LAYOUT / MATRIX / OTHER / ENTITY
  "hasHeader": true,                             // if true, the row at headerRowIndex is column NAMES, not values
  "headerRowIndex": 0,                           // 0-based; -1 when no header
  "pageTitle": "...",                            // optional; preserved into the output
  "url": "...",                                  // optional; preserved into the output
  "tableNum": 11,                                // optional; preserved into the output
  // ... any other metadata fields are also preserved
}
```

Other fields are passed through untouched on output.

### CSV folder (secondary)

Pass a directory containing one or more `.csv` files. Each CSV becomes one table. CSVs are read row-major and transposed to column-major. Note that **CSVs have no header metadata**, so the first row is treated as data — if your CSVs really have headers, strip them before running, or convert to JSONL.

---

## 4. Output files

The pipeline writes five files into `--output_folder`:
- `filtered_corpus.jsonl` (WP1)
- `coherence_distribution.png` (WP1)
- `threshold_sweep.txt` (WP1)
- `cooccurrence_index.pkl` (WP1, cached for re-runs)
- `candidates.jsonl` (WP2 — see section 6 for the schema)

Detailed descriptions:

### `filtered_corpus.jsonl`

The corpus with low-coherence columns removed, ready as input to WP2 (column-pair extraction). One JSON object per line, mirroring the input schema with two added fields:

```jsonc
{
  "relation": [["united states","canada"], ["usa","can"]],   // surviving columns only
  "coherence_scores": [0.78, 0.82],                           // one score per surviving column
  "rejected_column_indices": [2, 4],                          // original indices that were dropped
  "tableType": "RELATION",
  "pageTitle": "...",                                         // metadata preserved verbatim
  "url": "...",
  // ...
}
```

Tables that lose all of their columns are not written at all.

**How to read:** if a row is in this file, every surviving column passed the threshold. The `rejected_column_indices` list tells you which original-relation positions were filtered out, so you can map the surviving columns back to the original table layout if you need to.

### `coherence_distribution.png`

Histogram of all column coherence scores, with a red dashed vertical line at `--threshold`.

**How to read:**
- **Bimodal distribution (two humps)** — the threshold is doing real work. Coherent columns cluster on the right (near +1), incoherent ones on the left (near 0 or below).
- **Single hump near +1** — most of the corpus is already clean. PMI alone may not be filtering much. This is what you'll see on `data/sample.json`: the WDC RELATION tables are mostly already coherent.
- **Single hump near 0** — values aren't co-occurring enough for NPMI to register signal. Usually means corpus is too small or values too unique. You probably need a larger corpus.
- **Mass to the left of the threshold line** — those are the columns being dropped. Rough sanity check on whether the threshold is set sensibly for your data.

### `threshold_sweep.txt`

Text table showing how many columns survive at each of {0.1, 0.2, 0.3, 0.4, 0.5}. Useful for picking a threshold without re-running.

```
Threshold | Kept | Removed | Kept %
----------+------+---------+-------
   0.1    | 11891 |    65   | 99.5%
   0.2    | 11811 |   145   | 98.8%
   0.3    | 11609 |   347   | 97.1%
   0.4    | 11019 |   937   | 92.2%
   0.5    | 10207 |  1749   | 85.4%
```

**How to read:**
- The Kept count must monotonically decrease as threshold rises (basic sanity check).
- The "right" threshold is wherever the slope changes — that's the boundary between coherent-by-most-definitions columns and borderline ones. Eyeball the histogram next to this table.
- Default `0.3` is the paper's suggested operating point. Use lower thresholds (0.1–0.2) to be more permissive, higher (0.4–0.5) to be stricter. There's no objectively-right answer; it's a tunable knob.

### `cooccurrence_index.pkl`

Pickled tuple `(cooccurrence: dict[(str,str), int], value_count: dict[str, int], total_columns: int)`. This is the global statistic over the whole corpus. You don't need to read it directly — it's there so re-runs skip the expensive index-building step.

If you want to peek:

```python
import pickle

with open("../output/cooccurrence_index.pkl", "rb") as f:
    cooc, vc, N = pickle.load(f)
print(f"Corpus had {N} columns, {len(vc)} unique values, {len(cooc)} unique value pairs")
```

---

## 5. Interpreting `candidates.jsonl`

Each line of `output/candidates.jsonl` is one **candidate mapping** — an
ordered column pair `(left, right)` from a single source table whose values
satisfy approximate functional dependency.

### Schema

```jsonc
{
  "pairs": [["united states", "usa"],
            ["canada", "can"],
            ["japan", "jpn"]],          // deduplicated (left, right) value pairs
                                         // — one entry per distinct left value,
                                         //    paired with its most common right.
  "theta": 0.971,                       // approximate-FD score; always >= --theta.
  "row_count": 7,                       // total non-empty rows used.
  "covered_rows": 6,                    // rows in the witness subset (where left
                                         //   maps to its most common right).
  "source_table_index": 0,              // line number in filtered_corpus.jsonl.
  "left_column_index": 0,               // index into the filtered table's relation.
  "right_column_index": 1,
  "source_metadata": {                  // pass-through metadata from WP1 output.
    "pageTitle": "...",
    "url": "...",
    "tableType": "RELATION",
    "tableNum": 11
  }
}
```

### How to read θ

θ is the fraction of rows that fit a clean FD (one left → one right). It's
in `[--theta, 1.0]` for any candidate that survived filtering.

| θ value | Meaning |
|---|---|
| 1.0 | Strict FD — every left maps to exactly one right. Cleanest mappings. |
| 0.99 | One or two row-level inconsistencies. Often real, with name ambiguity. |
| 0.95–0.99 | Moderate ambiguity — several lefts map to multiple rights, but a clear majority winner exists. |
| <0.95 | Rejected at the default threshold. |

### Inspecting candidates

**Top 20 candidates by θ:**
```bash
.venv/bin/python -c "
import json
rows = []
with open('output/candidates.jsonl') as f:
    for line in f:
        c = json.loads(line)
        rows.append((c['theta'], c['pairs'][:3], c['source_metadata'].get('pageTitle', '')[:50]))
rows.sort(reverse=True)
for theta, pairs, title in rows[:20]:
    print(f'{theta:.3f}  {pairs}  | {title}')
"
```

**How many candidates per source table:**
```bash
.venv/bin/python -c "
import json
from collections import Counter
counts = Counter()
with open('output/candidates.jsonl') as f:
    for line in f:
        c = json.loads(line)
        counts[c['source_table_index']] += 1
print(f'Tables contributing candidates: {len(counts)}')
print(f'Mean candidates per table: {sum(counts.values())/len(counts):.2f}')
print(f'Max candidates from one table: {max(counts.values())}')
"
```

**Browse one specific candidate:**
```bash
.venv/bin/python -c "
import json
with open('output/candidates.jsonl') as f:
    line = f.readline()
print(json.dumps(json.loads(line), indent=2))
"
```

### Choosing θ

The paper specifies θ ≥ 0.95 as the operating point — it accepts mappings
with mild name ambiguity (Portland → Oregon vs Portland → Maine) while
rejecting tables that are clearly not mappings. Lower θ (e.g. 0.85) keeps
more borderline cases at the cost of more spurious candidates. Higher θ
(e.g. 0.99) accepts only near-strict FD.

Re-run `main.py` with a different `--theta` to compare candidate counts.
The cached `cooccurrence_index.pkl` is reused, so only Stages 3-5 re-run
(~7s on the sample corpus).

---

## 6. Interpreting coherence scores

Each column gets a single number in `[-1.0, +1.0]`:

| Score range | What it means |
|---|---|
| `+0.8` to `+1.0` | Highly coherent. Values strongly co-occur with each other across the corpus. Examples: country lists, ISO code columns, stock ticker columns. |
| `+0.4` to `+0.8` | Coherent but with some weakly-related values. Most "real" mapping columns land here. |
| `+0.0` to `+0.4` | Borderline. Some pairs co-occur, others don't. Could be a column where one value is unusually frequent in many contexts. |
| `-0.4` to `+0.0` | Likely incoherent. Values appear together less than chance would predict. |
| `-1.0` to `-0.4` | Strongly incoherent. The values explicitly avoid each other in the corpus. |

The score is the **mean NPMI over all unordered pairs of distinct values in the column**. So a column with values `[a, b, c]` produces three pairs `(a,b)`, `(a,c)`, `(b,c)`, and the column's score is the average of those three NPMI values.

**Key intuition:** NPMI(u, v) compares how often u and v actually co-occur to how often they would co-occur if they were independent. A coherent column is one where the values are "the same kind of thing" — they show up together in lots of other columns across the corpus, so their joint probability is way higher than the product of their marginal probabilities, so NPMI is near +1.

---

## 7. Running tests

```bash
python -m pytest -v
```

50 tests covering:
- Value normalization (`clean_value`)
- JSONL and CSV loaders (header strip, type filter, transposition, metadata preservation)
- Index construction (sorted-pair keys, set-dedup per column, total column count)
- PMI/NPMI math (perfect co-occurrence → +1, never co-occur → -1, symmetry, range clipping)
- Coherence ranking (coherent column outranks garbage column)
- Filtering (partition above/below threshold, rebuilt corpus drops empty tables, JSONL schema)
- Reporting (histogram file written, threshold sweep table)
- End-to-end smoke test (subprocess invocation of `main.py` produces all four artifacts)

The tests use a hand-designed synthetic corpus in `conftest.py` — see [WALKTHROUGH.md](WALKTHROUGH.md) for why the corpus is structured the way it is.

---

## 8. Troubleshooting

**Stage 2 takes a long time on a fresh corpus.** That's the index build — quadratic in unique values per column, linear in number of columns. For ~12k columns and ~67k unique values it takes a few seconds; for 100M tables it would need MapReduce. Use `--rebuild_index` only when you actually need to.

**All columns kept (or all removed) at threshold 0.3.** Check the histogram. If everything clusters on one side, the threshold is wrong for your corpus, not the code. Look at the threshold_sweep.txt to find a more meaningful cutoff.

**`ModuleNotFoundError: No module named 'pandas'` (or matplotlib).** Your `python` is the system one, not the venv. Either activate `.venv` or use `.venv/bin/python` directly.

**Pipeline runs but `filtered_corpus.jsonl` is empty.** Either every table dropped to zero kept columns (check the threshold), or `tableType` filtering excluded everything (try `--table_types RELATION,ENTITY,OTHER` or just `--table_types ""` after lowering the strictness).

**Wrong scores on real data.** Check that `hasHeader` and `headerRowIndex` are set correctly in your JSONL. Including the header row as data inflates value counts for column names and skews PMI.
