# WP1 — PMI Coherence Filtering: Design Spec

**Date:** 2026-05-05
**Project:** Auto-Map (re-implementation of Wang & He, SIGMOD 2017, *Synthesizing Mapping Relationships Using Table Corpus*)
**Scope:** Section 3.1 of the paper — Column Filtering by PMI. WP2 (Section 3.2, FD-based column-pair filtering) and beyond are out of scope.

---

## 1. Goal

Given a corpus of tables (web tables in JSONL form, or CSVs in a folder), produce a filtered corpus in which every surviving column has a coherence score (mean pairwise NPMI of its distinct values) at or above a threshold τ.

The output is the input to WP2 (column-pair extraction).

## 2. Inputs and outputs

### Input
- **Primary:** a JSON Lines file. Each line is a web table object with at minimum: `relation` (column-major `string[][]`), `tableType`, `hasHeader`, `headerRowIndex`. May also carry `pageTitle`, `url`, `tableNum`, etc.
- **Secondary:** a folder of CSVs. Each CSV is row-major; the loader transposes to columns. No metadata.
- The loader auto-dispatches on path: `*.json`/`*.jsonl` → JSONL backend; directory → CSV backend.

### Output
- `{output_folder}/filtered_corpus.jsonl` — filtered corpus, mirrors input schema with rejected columns removed.
- `{output_folder}/coherence_distribution.png` — histogram with threshold line.
- `{output_folder}/threshold_sweep.txt` — kept-vs-removed counts at thresholds {0.1, 0.2, 0.3, 0.4, 0.5}.
- `{index_path}` — pickled `(cooccurrence, value_count, total_columns)` for re-runs.

## 3. Math (paper-faithful)

For value pair `(u, v)` over a corpus of `N` columns where `C(u)` denotes the set of columns containing `u`:

```
p(u)    = |C(u)| / N
p(u,v)  = |C(u) ∩ C(v)| / N
PMI(u,v) = log( p(u,v) / (p(u) * p(v)) )
NPMI(u,v) = PMI(u,v) / -log(p(u,v))             ∈ [-1, +1]

Coherence(C) = mean over all unordered pairs (v_i, v_j), i<j, of NPMI(v_i, v_j)
             = sum / C(|unique(C)|, 2)
```

A column is **kept** iff `Coherence(C) >= τ`. Default τ = 0.3.

### Edge cases

| Case | Return | Reason |
|---|---|---|
| `p(u,v) = 0` | NPMI = -1 | Lower bound; "never co-occur" is maximally incoherent. |
| `p(u) = 0` or `p(v) = 0` | NPMI = -1 | Defensive; shouldn't occur in normal flow. |
| `p(u,v) = 1` (every column has both) | NPMI = +1 | `-log p(u,v) = 0` → div-by-zero guarded. |
| `math` domain error | NPMI = -1 | try/except. |
| Floating-point drift outside `[-1,+1]` | clipped | Final clip before return. |
| Column with 1 unique value | filtered upstream by loader | Cannot compute pairwise score. |

NPMI is symmetric: `npmi(u,v) == npmi(v,u)`. Pairs in the cooccurrence index are stored sorted (`tuple(sorted([u,v]))`) so `(germany, france)` and `(france, germany)` collide.

### Why "unique values per column" matters
The index counts each distinct value once per column (set, not multiset). If "USA" appears 50 times in one column, `value_count["USA"]` increments by 1 (not 50). This is what makes `p(u) = |C(u)| / N` correct as defined.

## 4. Architecture

```
JSONL/CSV → data_loader → corpus (List[Table]) ─┐
                                                ├→ cooccurrence_index → Index ─┐
                                                │                              ├→ npmi (score_corpus) → ScoredColumns ─┐
                                                │                              │                                       ├→ filter → kept/removed
                                                │                              │                                       │     │
                                                │                              │                                       │     ├→ rebuild_filtered_corpus → JSONL
                                                │                              │                                       │     ├→ plot_coherence_distribution → PNG
                                                │                              │                                       │     └→ threshold_sweep → txt
                                                │                              │                                       │
                                                │                              └→ save_index ──── (pickle, cached) ────┘
                                                │
                                                └→ corpus_summary (stdout)
```

Five modules + tests, plus `main.py` to orchestrate.

## 5. Module specs

### 5.1 `data_loader.py`

```python
Table = Tuple[Dict[str, Any], List[List[str]]]   # (metadata, columns)

def clean_value(value: Any) -> str:
    """Strip whitespace, lowercase, collapse internal whitespace.
    None/NaN/non-string → ''. Returned values are never None."""

def load_corpus(
    path: str,
    *,
    table_types: Iterable[str] = ("RELATION",),
    strip_headers: bool = True,
) -> List[Table]:
    """Auto-dispatch on path. JSONL: filter to table_types, drop header row
    when hasHeader=True. CSV folder: each .csv → one Table; metadata is {}.

    All values run through clean_value. Drop columns with <2 unique values.
    Drop empty columns."""

def _load_jsonl(path: str, table_types, strip_headers) -> List[Table]: ...
def _load_csv_folder(path: str) -> List[Table]: ...

def corpus_summary(corpus: List[Table]) -> None:
    """Print: # tables, # columns, # unique values, avg cols/table, time."""
```

**Rules:**
- Skip empty columns.
- Skip columns with <2 unique values.
- Strip whitespace, lowercase.
- For JSONL: drop tables whose `tableType` is not in `table_types`. Default `("RELATION",)` only.
- For JSONL with `hasHeader=True` and `headerRowIndex=k`, exclude index `k` of each column.
- Encoding: `utf-8` with `errors='ignore'` on file reads.

### 5.2 `cooccurrence_index.py`

```python
Index = Tuple[Dict[Tuple[str, str], int], Dict[str, int], int]

def build_cooccurrence_index(corpus: List[Table]) -> Index:
    """Single pass. For each column, take set(column), then:
       - increment value_count[v] for each v in set
       - increment cooccurrence[(min(u,v), max(u,v))] for each pair in combinations(set, 2)
       - increment total_columns
    Returns (cooccurrence, value_count, total_columns)."""

def save_index(index: Index, filepath: str) -> None:
    """Pickle. Pickle (not JSON) because tuple keys are first-class."""

def load_index(filepath: str) -> Index: ...

def index_summary(index: Index) -> None:
    """Print top-20 most common co-occurring pairs."""
```

**Rules:**
- Use `defaultdict(int)`.
- Pair keys are sorted: `tuple(sorted([u, v]))`.
- No size cap on unique values per column (paper-faithful).

### 5.3 `npmi.py`

```python
def compute_pmi(u: str, v: str, index: Index) -> float: ...
def compute_npmi(u: str, v: str, index: Index) -> float:
    """Returns ∈ [-1, +1]. Edge cases per Section 3."""
def compute_coherence(column: List[str], index: Index) -> float:
    """Mean NPMI over all C(|unique(column)|, 2) pairs."""

ScoredColumn = Tuple[int, int, List[str], float]

def score_corpus(corpus: List[Table], index: Index) -> List[ScoredColumn]:
    """For every column, emit (table_idx, col_idx, column_values, score)."""

def test_npmi(index: Index) -> None:
    """Sanity checks: top pair has positive NPMI; symmetry; range."""
```

**Rules:**
- All pairs computed (no sampling).
- Final clip to `[-1, +1]`.
- Handle math errors with try/except, returning -1.

### 5.4 `filter.py`

```python
def filter_corpus(
    scored: List[ScoredColumn],
    threshold: float = 0.3,
) -> Tuple[List[ScoredColumn], List[ScoredColumn]]:
    """Returns (kept, removed)."""

def rebuild_filtered_corpus(
    corpus: List[Table],
    kept: List[ScoredColumn],
) -> List[Tuple[Dict, List[List[str]], List[float], List[int]]]:
    """For each table, keep only the columns whose (table_idx, col_idx) is in
    kept. Drop tables with 0 surviving columns. Returns list of
    (metadata, kept_columns, kept_scores, rejected_column_indices)."""

def save_filtered_corpus(filtered, output_path: str) -> None:
    """JSONL. One object per line, mirroring input schema with rejected
    columns removed and added fields:
      - 'coherence_scores': List[float]   (one per kept column)
      - 'rejected_column_indices': List[int]   (original indices)"""

def filtering_report(kept, removed) -> None:
    """Print:
      - Total columns before/after
      - Percentage removed
      - 5 example removed columns with scores
      - 5 example kept columns with scores"""

def plot_coherence_distribution(scored, threshold, output_path: str) -> None:
    """Histogram, x ∈ [-1, +1], red dashed vertical line at threshold."""

def threshold_sweep(
    scored,
    thresholds: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
) -> None:
    """Print kept/removed/percentage table for each threshold."""
```

### 5.5 `main.py`

CLI:
```
--corpus_path     path to JSONL file or folder of CSVs
--output_folder   where to write outputs
--threshold       coherence threshold, default 0.3
--index_path      path to save/load co-occurrence index, default output/cooccurrence_index.pkl
--rebuild_index   force index rebuild even if cached
--table_types     comma-separated, default 'RELATION' (JSONL only)
```

**Behavior:**
- If `index_path` exists and `rebuild_index` not set, load instead of rebuilding.
- Print four stage banners (`[Stage k/4]`), each with stats and elapsed time.
- After Stage 3, print highest- and lowest-scoring columns for sanity.
- After Stage 4, print final summary and total elapsed.

## 6. Output schemas

### `filtered_corpus.jsonl`
```jsonc
{
  "relation": [["united states", "canada"], ["usa", "can"]],
  "coherence_scores": [0.78, 0.82],
  "rejected_column_indices": [2, 4],
  "pageTitle": "...",
  "url": "...",
  "tableType": "RELATION",
  "hasHeader": true,
  "headerRowIndex": 0,
  "tableNum": 11
  // ... all other input metadata fields preserved verbatim
}
```

Tables with 0 surviving columns are omitted.

### `coherence_distribution.png`
- Histogram of all column scores
- x ∈ [-1, +1], y = column count
- Vertical red dashed line at `--threshold`
- Title: `Column coherence distribution (N={total_columns}, threshold={τ})`

### `threshold_sweep.txt`
```
Threshold | Kept | Removed | Kept %
----------+------+---------+-------
   0.1    | 4823 |  1120   | 81.2%
   0.2    | 4101 |  1842   | 69.0%
   0.3    | 3204 |  2739   | 53.9%
   0.4    | 2110 |  3833   | 35.5%
   0.5    | 1188 |  4755   | 20.0%
```

### `cooccurrence_index.pkl`
Python pickle of `(cooccurrence_dict, value_count_dict, total_columns_int)`.

## 7. Testing

`tests/test_wp1.py` — pytest. Synthetic corpus is constructed in-code (no fixture files):

```python
SYNTHETIC = [
    [["united states", "canada", "japan"], ["usa", "can", "jpn"]],
    [["united states", "canada", "germany"], ["usa", "can", "deu"]],
    [["japan", "germany", "france"], ["jpn", "deu", "fra"]],
    [["msft", "aapl", "googl"], ["microsoft", "apple", "alphabet"]],
    [["msft", "aapl"], ["microsoft", "apple"]],
    [["2024-01-01", "hello world", "83.5%", "the matrix", "blue"]],
]
```

Cases:
1. `test_index_construction` — pair count and `total_columns` match hand-computed values.
2. `test_npmi_known_pairs` — co-occurring pair NPMI > 0.6; never-co-occurring pair = -1.
3. `test_coherence_ranking` — country/iso column outranks garbage column; garbage < 0.
4. `test_filter_partition` — at τ=0.3 country/iso/ticker survive, garbage doesn't.
5. `test_npmi_clipping` — every NPMI ∈ [-1, +1].
6. `test_npmi_symmetry` — `npmi(u,v) == npmi(v,u)` for 100 random pairs from the index.
7. `test_index_pickle_roundtrip` — save → load gives identical index.
8. `test_filtered_jsonl_schema` — every output line has `relation`, `coherence_scores` (length matches `relation` width), `rejected_column_indices`.
9. `test_empty_table_dropped` — table with 0 surviving columns absent from output.
10. `test_header_stripping` — JSONL loader with `hasHeader=true` excludes header row.

No mocks. Tests run real math on real synthetic data.

**Integration sanity:** `main.py` itself is the integration test — running on `data/sample.json` should produce a histogram with a clear bimodal distribution (coherent vs incoherent columns).

## 8. Constraints

- Python 3.8+
- Allowed libs: pandas, numpy, matplotlib, collections, itertools, math, json, pickle, os, sys, argparse, random, time, typing
- Every public function: type hints + docstring
- Every module: module-level docstring
- Print progress at each stage

## 9. Deviations from `claude/wp1_prompt.md`

Captured in `claude/deviations.md`. Summary:

| # | Prompt | Spec | Why |
|---|---|---|---|
| 1 | CSV folder input only | JSONL primary + CSV folder secondary, auto-dispatch | Sample data is JSONL web-table corpus |
| 2 | Skip cols with >100 unique values | No size cap | Paper relies on PMI to filter |
| 3 | Sample 50 pairs for cols with >50 unique values | All pairs | Paper computes exact average |
| 4 | No table-type filtering | RELATION-only by default | Paper assumes relational corpus |
| 5 | No header handling | Strip header row when `hasHeader=true` | Otherwise headers pollute the index |
| 6 | `load_table(filepath)` as top-level | Internal helper | Auto-dispatch loader is cleaner |
| 7 | One CSV per filtered table | One `filtered_corpus.jsonl` | Preserves metadata, avoids tiny-files churn |
| 8 | JSON for index (pickle as fallback) | Pickle as primary | Tuple keys + speed |
| 9 | 20 generated CSV fixtures | In-code synthetic mini-corpus in tests | Real data already present |
| 10 | `--corpus_folder` flag | `--corpus_path` flag | Path can be file or folder |

Math, file structure (5 modules + main), threshold default (0.3), threshold sweep, histogram, stage timing, progress prints — unchanged from prompt.

## 10. Out of scope

- WP2 (Section 3.2): column-pair generation + FD-based filtering.
- WP3 (Section 4): table synthesis via positive-compatibility + FD-conflict graph optimization.
- WP4 (Section 5): conflict resolution.
- Streaming / out-of-core processing (sample fits in memory; future refactor).
- Distributed execution (paper uses MapReduce; we run in-process).

## 11. Deliverables

```
data_loader.py
cooccurrence_index.py
npmi.py
filter.py
main.py
tests/test_wp1.py
claude/deviations.md
README.md   (run command, architecture sketch, output description)
```

Run command:
```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/ \
               --threshold 0.3 \
               --index_path output/cooccurrence_index.pkl
```
