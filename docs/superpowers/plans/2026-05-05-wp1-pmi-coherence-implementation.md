# WP1 PMI Coherence Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-implement Section 3.1 of Wang & He (SIGMOD 2017) — PMI-based column coherence filtering — as a runnable pipeline that takes a JSONL web-table corpus or a folder of CSVs and writes a filtered JSONL corpus, a coherence histogram, and a threshold sweep summary.

**Architecture:** Five flat modules (`data_loader.py`, `cooccurrence_index.py`, `npmi.py`, `filter.py`, `main.py`) loaded into one in-memory pipeline. The corpus is read once, the global `(cooccurrence, value_count, total_columns)` index is built in a single pass and pickled for re-runs, every column is scored (mean pairwise NPMI of its distinct values), columns above the threshold are kept, and the filtered corpus is emitted with metadata preserved.

**Tech Stack:** Python 3.8+, pandas, numpy, matplotlib, pytest, plus stdlib (`collections`, `itertools`, `math`, `json`, `pickle`, `os`, `sys`, `argparse`, `random`, `time`, `typing`).

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-05-wp1-pmi-coherence-design.md`
- Deviations: `claude/deviations.md`
- Source paper: `papers/automap.pdf` (Section 3.1)
- Original prompt: `claude/wp1_prompt.md`
- Sample input: `data/sample.json` (JSONL, one web table per line)

**Important commit convention:** Do NOT include any `Co-Authored-By: Claude ...` trailer on commits. Plain commit messages only.

---

## File map

| Path | Status | Purpose |
|---|---|---|
| `requirements.txt` | create | Pin runtime + test deps |
| `conftest.py` | create | pytest rootdir + shared `synthetic_corpus` fixture |
| `tests/test_wp1.py` | create, grow per task | All unit tests for WP1 |
| `data_loader.py` | create | `clean_value`, `_load_jsonl`, `_load_csv_folder`, `load_corpus`, `corpus_summary` |
| `cooccurrence_index.py` | create | `build_cooccurrence_index`, `save_index`, `load_index`, `index_summary` |
| `npmi.py` | create | `compute_pmi`, `compute_npmi`, `compute_coherence`, `score_corpus`, `test_npmi` |
| `filter.py` | create | `filter_corpus`, `rebuild_filtered_corpus`, `save_filtered_corpus`, `filtering_report`, `threshold_sweep`, `plot_coherence_distribution` |
| `main.py` | create | CLI orchestration with `[Stage k/4]` progress + timings |
| `README.md` | overwrite | How to run and what the outputs mean |

`claude/deviations.md` already exists; do not modify.

---

## Task 0: Project skeleton

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py`
- Create: `tests/test_wp1.py` (placeholder)

- [ ] **Step 1: Create `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
pytest>=7.4
```

- [ ] **Step 2: Create `conftest.py` at repo root**

```python
"""pytest rootdir marker. Putting this file at the repo root makes pytest
add the root to sys.path so tests can `import data_loader`, `import npmi`,
etc. without a package layout."""

import pytest


@pytest.fixture
def synthetic_corpus():
    """Synthetic mini-corpus for unit tests.

    Five tables of country/iso pairs and ticker/company pairs (coherent),
    plus one table with a single garbage column (incoherent). Designed so
    that coherent columns score near +1 and the garbage column scores -1.
    """
    return [
        ({}, [["united states", "canada", "japan"], ["usa", "can", "jpn"]]),
        ({}, [["united states", "canada", "germany"], ["usa", "can", "deu"]]),
        ({}, [["japan", "germany", "france"], ["jpn", "deu", "fra"]]),
        ({}, [["msft", "aapl", "googl"], ["microsoft", "apple", "alphabet"]]),
        ({}, [["msft", "aapl"], ["microsoft", "apple"]]),
        ({}, [["2024-01-01", "hello world", "83.5%", "the matrix", "blue"]]),
    ]
```

- [ ] **Step 3: Create empty `tests/test_wp1.py`**

```python
"""Unit tests for WP1 (PMI coherence filtering).

Tests grow incrementally as each module is implemented per the plan."""
```

- [ ] **Step 4: Install deps and verify pytest discovers the test module**

```bash
pip install -r requirements.txt
python -m pytest --collect-only -q
```
Expected: `0 tests collected` exit 5 — pytest finds `tests/test_wp1.py` but no test functions yet.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt conftest.py tests/test_wp1.py
git commit -m "scaffold: requirements, pytest rootdir, empty test module"
```

---

## Task 1: `clean_value`

**Files:**
- Create: `data_loader.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from data_loader import clean_value


def test_clean_value_strips_and_lowercases():
    assert clean_value("  Germany  ") == "germany"


def test_clean_value_collapses_internal_whitespace():
    assert clean_value("New   York") == "new york"


def test_clean_value_handles_none():
    assert clean_value(None) == ""


def test_clean_value_handles_non_string():
    assert clean_value(42) == "42"
    assert clean_value(3.14) == "3.14"


def test_clean_value_handles_nan():
    import math
    assert clean_value(math.nan) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: 5 failures with `ModuleNotFoundError: No module named 'data_loader'`.

- [ ] **Step 3: Implement `clean_value`**

Create `data_loader.py`:

```python
"""Loader for table corpora.

Supports two input formats, auto-dispatched on path:
- JSON Lines file (`.json` / `.jsonl`): one web-table object per line.
- Folder of CSVs: each `.csv` becomes one table.

The loader produces a list of `Table = (metadata: dict, columns: list[list[str]])`
tuples where columns are already normalized (strip + lowercase + whitespace
collapse) and short columns (<2 unique values, empty) are dropped.
"""

from __future__ import annotations

import math
import re
from typing import Any

_WS_RE = re.compile(r"\s+")


def clean_value(value: Any) -> str:
    """Normalize a single cell value.

    - None / NaN  -> ""
    - Any value cast to str, then stripped, lowercased, and internal
      whitespace runs collapsed to a single space.

    Returns a string (never None).
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip().lower()
    return _WS_RE.sub(" ", s)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add data_loader.py tests/test_wp1.py
git commit -m "data_loader: clean_value with whitespace + None/NaN handling"
```

---

## Task 2: JSONL backend `_load_jsonl`

**Files:**
- Modify: `data_loader.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
import json
from data_loader import _load_jsonl


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_jsonl_loader_reads_relation_as_columns(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["United States", "Canada"], ["USA", "CAN"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    assert len(corpus) == 1
    metadata, columns = corpus[0]
    assert columns == [["united states", "canada"], ["usa", "can"]]


def test_jsonl_loader_strips_header_row(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["Country", "USA", "Canada"], ["Code", "USA", "CAN"]],
        "tableType": "RELATION",
        "hasHeader": True,
        "headerRowIndex": 0,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    # Header row removed: "Country"/"Code" gone from each column
    assert columns == [["usa", "canada"], ["usa", "can"]]


def test_jsonl_loader_filters_table_types(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [
        {"relation": [["a", "b"], ["c", "d"]], "tableType": "LAYOUT",
         "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["a", "b"], ["c", "d"]], "tableType": "RELATION",
         "hasHeader": False, "headerRowIndex": -1},
    ])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    assert len(corpus) == 1


def test_jsonl_loader_drops_columns_below_2_unique(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["x", "x", "x"], ["a", "b", "c"]],  # first column has 1 unique
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    assert len(columns) == 1
    assert columns[0] == ["a", "b", "c"]


def test_jsonl_loader_drops_empty_columns(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["", "", ""], ["a", "b", "c"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    assert len(columns) == 1


def test_jsonl_loader_preserves_metadata(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["a", "b"], ["c", "d"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
        "pageTitle": "Hello",
        "url": "http://example.com",
        "tableNum": 7,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    assert metadata["pageTitle"] == "Hello"
    assert metadata["url"] == "http://example.com"
    assert metadata["tableNum"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: ImportError / failures referencing `_load_jsonl`.

- [ ] **Step 3: Implement `_load_jsonl`**

Append to `data_loader.py`:

```python
import json
from typing import Iterable, List, Tuple, Dict

Table = Tuple[Dict[str, Any], List[List[str]]]


def _load_jsonl(
    path: str,
    table_types: Iterable[str] = ("RELATION",),
    strip_headers: bool = True,
) -> List[Table]:
    """Load tables from a JSON Lines file.

    Each line is a JSON object with at minimum:
      - `relation`: column-major list[list[Any]]
      - `tableType`: string
      - `hasHeader`: bool
      - `headerRowIndex`: int  (-1 when no header)

    Tables whose `tableType` is not in `table_types` are skipped. When
    `strip_headers` is True and `hasHeader` is True, the row at
    `headerRowIndex` is removed from every column. Values pass through
    `clean_value`. Columns that are empty or have <2 unique values are
    dropped. The returned metadata dict is the input record minus the
    `relation` field (which is replaced by the cleaned columns).
    """
    keep = set(table_types)
    out: List[Table] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("tableType") not in keep:
                continue
            relation = rec.get("relation") or []
            if strip_headers and rec.get("hasHeader") and rec.get("headerRowIndex", -1) >= 0:
                hri = rec["headerRowIndex"]
                relation = [
                    [v for i, v in enumerate(col) if i != hri]
                    for col in relation
                ]
            cleaned: List[List[str]] = []
            for col in relation:
                vals = [clean_value(v) for v in col]
                vals = [v for v in vals if v != ""]
                if len(set(vals)) < 2:
                    continue
                cleaned.append(vals)
            if not cleaned:
                continue
            metadata = {k: v for k, v in rec.items() if k != "relation"}
            out.append((metadata, cleaned))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass (Task 1 + Task 2 ones).

- [ ] **Step 5: Commit**

```bash
git add data_loader.py tests/test_wp1.py
git commit -m "data_loader: JSONL backend with header strip and type filter"
```

---

## Task 3: CSV backend `_load_csv_folder`

**Files:**
- Modify: `data_loader.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from data_loader import _load_csv_folder


def test_csv_folder_loads_and_transposes(tmp_path):
    csv_path = tmp_path / "table1.csv"
    csv_path.write_text("Country,Code\nUSA,USA\nCanada,CAN\nJapan,JPN\n", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    assert len(corpus) == 1
    metadata, columns = corpus[0]
    # CSV is read row-major then transposed; first row is treated as data,
    # which means "Country"/"Code" become regular values. CSV has no header
    # metadata so the loader cannot distinguish.
    assert columns[0] == ["country", "usa", "canada", "japan"]
    assert columns[1] == ["code", "usa", "can", "jpn"]


def test_csv_folder_loads_multiple_files(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("p,q\n5,6\n7,8\n", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    assert len(corpus) == 2


def test_csv_folder_skips_non_csv_files(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    assert len(corpus) == 1


def test_csv_folder_drops_short_columns(tmp_path):
    (tmp_path / "a.csv").write_text("a,b\nx,1\nx,2\n", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    metadata, columns = corpus[0]
    # First column ["a","x","x"] has 2 unique ("a","x"), kept.
    # Second column ["b","1","2"] has 3 unique, kept.
    assert len(columns) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: 4 new failures referencing `_load_csv_folder`.

- [ ] **Step 3: Implement `_load_csv_folder`**

Append to `data_loader.py`:

```python
import csv
import os


def _load_csv_folder(path: str) -> List[Table]:
    """Load every `.csv` file in `path` as a separate table.

    CSVs are row-major; we transpose to column-major. Values pass through
    `clean_value`. Columns that are empty or have <2 unique values are
    dropped. CSV files carry no metadata, so the returned metadata dict
    contains only `{"source": <filename>}` for traceability.
    """
    out: List[Table] = []
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(".csv"):
            continue
        fp = os.path.join(path, name)
        with open(fp, "r", encoding="utf-8", errors="ignore", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        width = max(len(r) for r in rows)
        columns_raw: List[List[str]] = [[] for _ in range(width)]
        for row in rows:
            for i in range(width):
                columns_raw[i].append(row[i] if i < len(row) else "")
        cleaned: List[List[str]] = []
        for col in columns_raw:
            vals = [clean_value(v) for v in col]
            vals = [v for v in vals if v != ""]
            if len(set(vals)) < 2:
                continue
            cleaned.append(vals)
        if cleaned:
            out.append(({"source": name}, cleaned))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add data_loader.py tests/test_wp1.py
git commit -m "data_loader: CSV folder backend with row->column transpose"
```

---

## Task 4: `load_corpus` dispatch + `corpus_summary`

**Files:**
- Modify: `data_loader.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from data_loader import load_corpus, corpus_summary


def test_load_corpus_dispatches_to_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["a", "b"], ["c", "d"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = load_corpus(str(p))
    assert len(corpus) == 1


def test_load_corpus_dispatches_to_csv_folder(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    corpus = load_corpus(str(tmp_path))
    assert len(corpus) == 1


def test_load_corpus_raises_on_missing_path(tmp_path):
    import pytest as _pt
    with _pt.raises(FileNotFoundError):
        load_corpus(str(tmp_path / "does_not_exist"))


def test_corpus_summary_runs(synthetic_corpus, capsys):
    corpus_summary(synthetic_corpus)
    captured = capsys.readouterr()
    assert "tables" in captured.out.lower()
    assert "columns" in captured.out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: 4 failures.

- [ ] **Step 3: Implement `load_corpus` and `corpus_summary`**

Append to `data_loader.py`:

```python
def load_corpus(
    path: str,
    *,
    table_types: Iterable[str] = ("RELATION",),
    strip_headers: bool = True,
) -> List[Table]:
    """Auto-dispatch on path: file -> JSONL backend, dir -> CSV-folder backend.

    Args:
        path: filesystem path. `*.json`/`*.jsonl` files use the JSONL
            backend; directories use the CSV-folder backend.
        table_types: JSONL only - which `tableType` values to keep.
        strip_headers: JSONL only - whether to drop the header row.

    Returns: list of (metadata, columns) tuples.
    """
    if os.path.isdir(path):
        return _load_csv_folder(path)
    if os.path.isfile(path):
        return _load_jsonl(path, table_types=table_types, strip_headers=strip_headers)
    raise FileNotFoundError(f"Corpus path does not exist: {path}")


def corpus_summary(corpus: List[Table]) -> None:
    """Print summary statistics for a loaded corpus to stdout."""
    n_tables = len(corpus)
    n_cols = sum(len(cols) for _, cols in corpus)
    unique_values: set = set()
    for _, cols in corpus:
        for col in cols:
            unique_values.update(col)
    avg_cols = n_cols / n_tables if n_tables else 0
    print(f"  Loaded {n_tables} tables, {n_cols} columns, {len(unique_values)} unique values")
    print(f"  Avg columns per table: {avg_cols:.2f}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add data_loader.py tests/test_wp1.py
git commit -m "data_loader: load_corpus dispatch and corpus_summary"
```

---

## Task 5: `build_cooccurrence_index`

**Files:**
- Create: `cooccurrence_index.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from cooccurrence_index import build_cooccurrence_index


def test_index_total_columns(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # Tables 0-4 contribute 2 cols each (10), table 5 contributes 1 col -> 11.
    assert N == 11


def test_index_value_count_dedupes_within_column(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # "united states" appears in column 0 of tables 0 and 1 only -> count 2.
    assert vc["united states"] == 2
    # "msft" appears in column 0 of tables 3 and 4 -> count 2.
    assert vc["msft"] == 2


def test_index_cooccurrence_uses_sorted_keys(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # ("canada", "united states") -> sorted key, present in tables 0 and 1.
    key = tuple(sorted(["united states", "canada"]))
    assert cooc[key] == 2
    # Reverse-order key should not exist.
    assert ("united states", "canada") not in cooc or key == ("canada", "united states")


def test_index_garbage_pairs_unique_to_one_column(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # ("hello world", "the matrix") only co-occurs in the garbage column.
    key = tuple(sorted(["hello world", "the matrix"]))
    assert cooc[key] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: failures referencing `cooccurrence_index`.

- [ ] **Step 3: Implement `build_cooccurrence_index`**

Create `cooccurrence_index.py`:

```python
"""Build a global co-occurrence index over a table corpus.

For each column in each table:
- value_count[v] increments by 1 for every distinct value v in the column
  (set, not multiset — a value present 50 times in one column counts once).
- cooccurrence[(min(u,v), max(u,v))] increments by 1 for every distinct
  pair (u, v) of values in the column.
- total_columns increments by 1.

The resulting `Index = (cooccurrence, value_count, total_columns)` lets us
compute p(u), p(v), p(u,v) as the fractions used in PMI.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

Index = Tuple[Dict[Tuple[str, str], int], Dict[str, int], int]


def build_cooccurrence_index(corpus) -> Index:
    """Single pass over the corpus building (cooccurrence, value_count, N)."""
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    value_count: Dict[str, int] = defaultdict(int)
    total_columns = 0
    for _metadata, columns in corpus:
        for col in columns:
            distinct = sorted(set(col))
            if len(distinct) < 2:
                continue
            total_columns += 1
            for v in distinct:
                value_count[v] += 1
            for u, v in combinations(distinct, 2):
                cooccurrence[(u, v)] += 1
    return dict(cooccurrence), dict(value_count), total_columns
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add cooccurrence_index.py tests/test_wp1.py
git commit -m "cooccurrence_index: build index from corpus (sorted-pair keys)"
```

---

## Task 6: Index persistence + summary

**Files:**
- Modify: `cooccurrence_index.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from cooccurrence_index import save_index, load_index, index_summary


def test_index_pickle_roundtrip(synthetic_corpus, tmp_path):
    original = build_cooccurrence_index(synthetic_corpus)
    p = tmp_path / "idx.pkl"
    save_index(original, str(p))
    loaded = load_index(str(p))
    assert loaded[0] == original[0]
    assert loaded[1] == original[1]
    assert loaded[2] == original[2]


def test_index_summary_runs(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    index_summary(idx)
    captured = capsys.readouterr()
    assert "Top" in captured.out or "top" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: ImportError / 2 failures.

- [ ] **Step 3: Implement persistence + summary**

Append to `cooccurrence_index.py`:

```python
import pickle


def save_index(index: Index, filepath: str) -> None:
    """Pickle the index. Pickle (not JSON) because tuple keys are first-class."""
    with open(filepath, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(filepath: str) -> Index:
    """Load a pickled index produced by `save_index`."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def index_summary(index: Index) -> None:
    """Print top-20 most common co-occurring pairs to stdout."""
    cooc, vc, N = index
    top = sorted(cooc.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print(f"  Processed {N} columns")
    print(f"  Found {len(cooc)} unique value pairs")
    print(f"  Top co-occurring pairs:")
    for (u, v), count in top:
        print(f"    ({u}, {v}): {count}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add cooccurrence_index.py tests/test_wp1.py
git commit -m "cooccurrence_index: pickle save/load and summary printing"
```

---

## Task 7: PMI and NPMI

**Files:**
- Create: `npmi.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
import math
from npmi import compute_pmi, compute_npmi


def test_npmi_perfect_co_occurrence(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    # "united states" and "usa" only appear together; in tables 0 and 1
    # both are in the same columns. Expect NPMI = +1.
    score = compute_npmi("united states", "usa", idx)
    assert score == 1.0


def test_npmi_never_co_occur(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    # "united states" never appears with "msft".
    score = compute_npmi("united states", "msft", idx)
    assert score == -1.0


def test_npmi_unknown_value_returns_minus_one(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    score = compute_npmi("united states", "this_value_is_not_in_corpus", idx)
    assert score == -1.0


def test_npmi_symmetry(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    cooc, vc, N = idx
    pairs = list(cooc.keys())[:50]
    for u, v in pairs:
        assert compute_npmi(u, v, idx) == compute_npmi(v, u, idx)


def test_npmi_clipping_range(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    cooc, vc, N = idx
    for u, v in list(cooc.keys())[:200]:
        s = compute_npmi(u, v, idx)
        assert -1.0 <= s <= 1.0


def test_pmi_returns_negative_inf_when_pair_missing(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    assert compute_pmi("united states", "msft", idx) == float("-inf")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: failures referencing `npmi`.

- [ ] **Step 3: Implement PMI and NPMI**

Create `npmi.py`:

```python
"""PMI / NPMI / coherence over a co-occurrence index.

Per Wang & He (SIGMOD 2017), Section 3.1:

    PMI(u, v)  = log( p(u, v) / ( p(u) * p(v) ) )
    NPMI(u, v) = PMI(u, v) / -log( p(u, v) )      ∈ [-1, +1]

with p(x) = |C(x)| / N and p(u, v) = |C(u) ∩ C(v)| / N over a corpus of
N columns. NPMI is clipped to [-1, +1] to absorb floating-point drift.

Edge cases:
- Pair never co-occurs   -> NPMI = -1.0
- Either value not in corpus -> NPMI = -1.0
- p(u, v) = 1 (in every column) -> NPMI = +1.0 (-log p = 0 guarded)
- Any math domain error  -> NPMI = -1.0
"""

from __future__ import annotations

import math
from typing import List, Tuple

from cooccurrence_index import Index


def _pair_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def compute_pmi(u: str, v: str, index: Index) -> float:
    """PMI of (u, v). Returns -inf when any probability is zero.

    Most callers should use `compute_npmi` instead; PMI alone is unbounded.
    """
    cooc, vc, N = index
    co = cooc.get(_pair_key(u, v), 0)
    cu = vc.get(u, 0)
    cv = vc.get(v, 0)
    if co == 0 or cu == 0 or cv == 0 or N == 0:
        return float("-inf")
    p_uv = co / N
    p_u = cu / N
    p_v = cv / N
    return math.log(p_uv / (p_u * p_v))


def compute_npmi(u: str, v: str, index: Index) -> float:
    """NPMI of (u, v), clipped to [-1, +1]."""
    cooc, vc, N = index
    co = cooc.get(_pair_key(u, v), 0)
    cu = vc.get(u, 0)
    cv = vc.get(v, 0)
    if co == 0 or cu == 0 or cv == 0 or N == 0:
        return -1.0
    p_uv = co / N
    if p_uv >= 1.0:
        return 1.0
    try:
        pmi = math.log(p_uv / ((cu / N) * (cv / N)))
        npmi = pmi / -math.log(p_uv)
    except (ValueError, ZeroDivisionError):
        return -1.0
    if math.isnan(npmi) or math.isinf(npmi):
        return -1.0
    return max(-1.0, min(1.0, npmi))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add npmi.py tests/test_wp1.py
git commit -m "npmi: PMI and NPMI with edge-case handling and [-1,+1] clip"
```

---

## Task 8: Coherence and corpus scoring

**Files:**
- Modify: `npmi.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from npmi import compute_coherence, score_corpus, ScoredColumn


def test_coherence_coherent_column_high(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    # First country column of table 0.
    s = compute_coherence(["united states", "canada", "japan"], idx)
    assert s > 0.5


def test_coherence_garbage_column_low(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    s = compute_coherence(
        ["2024-01-01", "hello world", "83.5%", "the matrix", "blue"], idx
    )
    # All pairs never co-occur outside this single column, so NPMI for each
    # pair is < +1 (specifically, computed value), but the column is the
    # only place these values appear together. Coherence should still be
    # *lower* than coherent columns and clearly < 0.3 threshold.
    assert s < 0.3


def test_coherence_coherent_outranks_garbage(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    coherent = compute_coherence(["united states", "canada", "japan"], idx)
    garbage = compute_coherence(
        ["2024-01-01", "hello world", "83.5%", "the matrix", "blue"], idx
    )
    assert coherent > garbage


def test_score_corpus_returns_one_per_column(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    total_cols = sum(len(cols) for _, cols in synthetic_corpus)
    assert len(scored) == total_cols


def test_score_corpus_tuple_shape(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    table_idx, col_idx, values, score = scored[0]
    assert isinstance(table_idx, int)
    assert isinstance(col_idx, int)
    assert isinstance(values, list)
    assert isinstance(score, float)


def test_score_corpus_indices_correct(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    # Find the entry for table 5 (garbage). It has only one column.
    table5 = [s for s in scored if s[0] == 5]
    assert len(table5) == 1
    assert table5[0][1] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: failures referencing `compute_coherence` / `score_corpus`.

- [ ] **Step 3: Implement coherence + score_corpus**

Append to `npmi.py`:

```python
from itertools import combinations

ScoredColumn = Tuple[int, int, List[str], float]


def compute_coherence(column: List[str], index: Index) -> float:
    """Mean NPMI over all distinct unordered value pairs of `column`.

    Returns -1.0 if the column has fewer than 2 unique values (degenerate;
    the loader normally filters these out, but compute_coherence is safe to
    call directly).
    """
    distinct = sorted(set(column))
    if len(distinct) < 2:
        return -1.0
    total = 0.0
    n = 0
    for u, v in combinations(distinct, 2):
        total += compute_npmi(u, v, index)
        n += 1
    return total / n


def score_corpus(corpus, index: Index) -> List[ScoredColumn]:
    """Score every column in the corpus.

    Returns a list of `(table_idx, col_idx, column_values, coherence_score)`
    tuples. `col_idx` is the column's position within the (already-loaded
    and pre-filtered) table, not its original position in `relation`.
    """
    out: List[ScoredColumn] = []
    for ti, (_metadata, columns) in enumerate(corpus):
        for ci, col in enumerate(columns):
            out.append((ti, ci, col, compute_coherence(col, index)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add npmi.py tests/test_wp1.py
git commit -m "npmi: coherence and corpus scoring"
```

---

## Task 9: `test_npmi` runtime sanity helper

**Files:**
- Modify: `npmi.py`
- Modify: `tests/test_wp1.py`

This is the runtime helper specified in the prompt — it prints a sanity check for a built index, used by `main.py`. Not a pytest test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wp1.py`:

```python
from npmi import test_npmi as npmi_sanity_check


def test_runtime_sanity_helper_prints(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    npmi_sanity_check(idx)
    captured = capsys.readouterr()
    assert "NPMI" in captured.out or "npmi" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: `ImportError: cannot import name 'test_npmi'`.

- [ ] **Step 3: Implement `test_npmi`**

Append to `npmi.py`:

```python
def test_npmi(index: Index) -> None:
    """Print sanity stats for an index. Verifies symmetry and range on a
    sample of pairs and prints the highest-ranked pair. Helper for `main.py`,
    not a pytest test."""
    cooc, vc, N = index
    if not cooc:
        print("  NPMI sanity: index is empty")
        return
    # Top pair by raw co-occurrence.
    top_pair, top_count = max(cooc.items(), key=lambda kv: kv[1])
    u, v = top_pair
    print(f"  NPMI sanity: top pair ({u}, {v}) count={top_count} npmi={compute_npmi(u, v, index):.3f}")
    # Symmetry on first 100 pairs.
    sym_ok = all(
        compute_npmi(u, v, index) == compute_npmi(v, u, index)
        for u, v in list(cooc.keys())[:100]
    )
    print(f"  NPMI sanity: symmetric on first 100 pairs: {sym_ok}")
    # Range check on first 200 pairs.
    in_range = all(
        -1.0 <= compute_npmi(u, v, index) <= 1.0
        for u, v in list(cooc.keys())[:200]
    )
    print(f"  NPMI sanity: in-range on first 200 pairs: {in_range}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add npmi.py tests/test_wp1.py
git commit -m "npmi: test_npmi runtime sanity helper"
```

---

## Task 10: `filter_corpus`

**Files:**
- Create: `filter.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from filter import filter_corpus


def test_filter_partitions_above_and_below_threshold(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    assert all(s >= 0.3 for _, _, _, s in kept)
    assert all(s < 0.3 for _, _, _, s in removed)
    assert len(kept) + len(removed) == len(scored)


def test_filter_garbage_column_removed(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    # Table 5 column 0 is the garbage column.
    removed_ids = {(ti, ci) for ti, ci, _, _ in removed}
    assert (5, 0) in removed_ids


def test_filter_country_columns_kept(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    kept_ids = {(ti, ci) for ti, ci, _, _ in kept}
    # Tables 0-2 are country/iso, both columns should survive.
    for ti in (0, 1, 2):
        assert (ti, 0) in kept_ids
        assert (ti, 1) in kept_ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: ImportError on `filter`.

- [ ] **Step 3: Implement `filter_corpus`**

Create `filter.py`:

```python
"""Threshold-based filtering, JSONL output, and reporting for WP1.

Inputs come from `npmi.score_corpus`; outputs are
- a partition of scored columns into kept/removed,
- a rebuilt corpus where each retained table contains only kept columns,
- a JSONL file mirroring the input schema with the rejected columns
  removed and added `coherence_scores` / `rejected_column_indices` fields,
- a coherence histogram (PNG) and a threshold sweep summary.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

from npmi import ScoredColumn


def filter_corpus(
    scored: List[ScoredColumn],
    threshold: float = 0.3,
) -> Tuple[List[ScoredColumn], List[ScoredColumn]]:
    """Partition scored columns at `threshold`. Returns (kept, removed)."""
    kept: List[ScoredColumn] = []
    removed: List[ScoredColumn] = []
    for entry in scored:
        if entry[3] >= threshold:
            kept.append(entry)
        else:
            removed.append(entry)
    return kept, removed
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add filter.py tests/test_wp1.py
git commit -m "filter: filter_corpus partitions scored columns at threshold"
```

---

## Task 11: `rebuild_filtered_corpus` + `save_filtered_corpus`

**Files:**
- Modify: `filter.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from filter import rebuild_filtered_corpus, save_filtered_corpus


def test_rebuild_keeps_only_surviving_columns(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    # Each filtered entry: (metadata, kept_columns, kept_scores, rejected_indices)
    for metadata, columns, scores, rejected in filtered:
        assert len(columns) == len(scores)
        assert all(isinstance(s, float) for s in scores)


def test_rebuild_drops_tables_with_zero_kept(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    # Table 5 (garbage column) has zero kept columns -> dropped.
    assert len(filtered) == 5


def test_rebuild_records_rejected_indices(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    # In synthetic_corpus tables 0-4, both columns survive,
    # so rejected_indices is empty for those tables.
    for metadata, columns, scores, rejected in filtered:
        assert rejected == []


def test_save_filtered_corpus_jsonl_schema(synthetic_corpus, tmp_path):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    out = tmp_path / "filtered.jsonl"
    save_filtered_corpus(filtered, str(out))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        rec = json.loads(line)
        assert "relation" in rec
        assert "coherence_scores" in rec
        assert "rejected_column_indices" in rec
        assert len(rec["relation"]) == len(rec["coherence_scores"])


def test_save_filtered_corpus_preserves_metadata(tmp_path):
    # Tiny corpus with metadata attached so we can verify it carries through.
    corpus = [
        ({"pageTitle": "X", "url": "http://x"},
         [["united states", "canada"], ["usa", "can"]]),
    ]
    # Build a small index that makes the columns coherent.
    idx_corpus = corpus + [
        ({}, [["united states", "canada"], ["usa", "can"]]),
        ({}, [["united states", "canada"], ["usa", "can"]]),
    ]
    idx = build_cooccurrence_index(idx_corpus)
    scored = score_corpus(corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(corpus, kept)
    out = tmp_path / "filtered.jsonl"
    save_filtered_corpus(filtered, str(out))
    rec = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["pageTitle"] == "X"
    assert rec["url"] == "http://x"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: 5 failures referencing `rebuild_filtered_corpus` / `save_filtered_corpus`.

- [ ] **Step 3: Implement rebuild + save**

Append to `filter.py`:

```python
FilteredTable = Tuple[Dict, List[List[str]], List[float], List[int]]


def rebuild_filtered_corpus(
    corpus,
    kept: List[ScoredColumn],
) -> List[FilteredTable]:
    """Reconstruct each table with only its surviving columns.

    Tables with zero surviving columns are dropped. Returns a list of
    (metadata, kept_columns, kept_scores, rejected_column_indices) tuples
    in the original table order.
    """
    by_table: Dict[int, List[Tuple[int, List[str], float]]] = {}
    for ti, ci, values, score in kept:
        by_table.setdefault(ti, []).append((ci, values, score))

    out: List[FilteredTable] = []
    for ti, (metadata, columns) in enumerate(corpus):
        kept_for_table = sorted(by_table.get(ti, []), key=lambda x: x[0])
        if not kept_for_table:
            continue
        kept_indices = {c[0] for c in kept_for_table}
        kept_columns = [c[1] for c in kept_for_table]
        kept_scores = [c[2] for c in kept_for_table]
        rejected = [i for i in range(len(columns)) if i not in kept_indices]
        out.append((dict(metadata), kept_columns, kept_scores, rejected))
    return out


def save_filtered_corpus(filtered: List[FilteredTable], output_path: str) -> None:
    """Write the filtered corpus as JSON Lines.

    Each record mirrors the input schema with two added fields:
      - `coherence_scores`: list[float], one per surviving column
      - `rejected_column_indices`: list[int], original column indices removed
    The `relation` field is replaced with the surviving columns only.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metadata, columns, scores, rejected in filtered:
            rec: Dict = dict(metadata)
            rec["relation"] = columns
            rec["coherence_scores"] = scores
            rec["rejected_column_indices"] = rejected
            f.write(json.dumps(rec) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add filter.py tests/test_wp1.py
git commit -m "filter: rebuild filtered corpus and write JSONL with metadata"
```

---

## Task 12: Reporting + threshold sweep + histogram plot

**Files:**
- Modify: `filter.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp1.py`:

```python
from filter import filtering_report, threshold_sweep, plot_coherence_distribution


def test_filtering_report_runs(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    filtering_report(kept, removed)
    out = capsys.readouterr().out
    assert "before" in out.lower()
    assert "after" in out.lower()


def test_threshold_sweep_runs(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    threshold_sweep(scored, thresholds=(0.1, 0.3, 0.5))
    out = capsys.readouterr().out
    assert "0.1" in out
    assert "0.3" in out
    assert "0.5" in out


def test_plot_coherence_distribution_writes_file(synthetic_corpus, tmp_path):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    out = tmp_path / "hist.png"
    plot_coherence_distribution(scored, threshold=0.3, output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: 3 failures referencing the new functions.

- [ ] **Step 3: Implement reporting, sweep, and plot**

Append to `filter.py`:

```python
def filtering_report(
    kept: List[ScoredColumn],
    removed: List[ScoredColumn],
) -> None:
    """Print the filtering report described in the spec."""
    total = len(kept) + len(removed)
    pct = (len(removed) / total * 100) if total else 0.0
    print(f"  Total columns before filtering: {total}")
    print(f"  Total columns after filtering: {len(kept)}")
    print(f"  Removed: {len(removed)} ({pct:.1f}%)")
    print(f"  Examples of removed columns:")
    for ti, ci, vals, score in sorted(removed, key=lambda x: x[3])[:5]:
        sample = vals[:5]
        more = "..." if len(vals) > 5 else ""
        print(f"    table {ti} col {ci} score={score:.3f} values={sample}{more}")
    print(f"  Examples of kept columns:")
    for ti, ci, vals, score in sorted(kept, key=lambda x: x[3], reverse=True)[:5]:
        sample = vals[:5]
        more = "..." if len(vals) > 5 else ""
        print(f"    table {ti} col {ci} score={score:.3f} values={sample}{more}")


def threshold_sweep(
    scored: List[ScoredColumn],
    thresholds: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    output_path: str | None = None,
) -> str:
    """Print and (optionally) save a kept/removed table over multiple thresholds.

    Returns the rendered table as a string for downstream callers.
    """
    total = len(scored)
    lines = ["Threshold | Kept | Removed | Kept %",
             "----------+------+---------+-------"]
    for t in thresholds:
        kept = sum(1 for _, _, _, s in scored if s >= t)
        removed = total - kept
        pct = (kept / total * 100) if total else 0.0
        lines.append(f"   {t:.1f}    | {kept:>4} |  {removed:>4}   | {pct:.1f}%")
    rendered = "\n".join(lines)
    print(rendered)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    return rendered


def plot_coherence_distribution(
    scored: List[ScoredColumn],
    threshold: float,
    output_path: str,
) -> None:
    """Save a histogram of coherence scores with a vertical threshold line.

    Uses a non-interactive matplotlib backend so the call works in headless
    environments (CI, no display).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = [s for _, _, _, s in scored]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=40, range=(-1.0, 1.0))
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"threshold={threshold}")
    ax.set_xlabel("Coherence (mean pairwise NPMI)")
    ax.set_ylabel("Column count")
    ax.set_title(
        f"Column coherence distribution (N={len(scored)}, threshold={threshold})"
    )
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add filter.py tests/test_wp1.py
git commit -m "filter: filtering_report, threshold_sweep, histogram plot"
```

---

## Task 13: `main.py` CLI orchestration

**Files:**
- Create: `main.py`
- Modify: `tests/test_wp1.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wp1.py`:

```python
import subprocess
import sys


def test_main_end_to_end_on_synthetic_jsonl(tmp_path):
    """Smoke test: run main.py against a tiny synthetic JSONL and verify
    the three expected output artifacts exist."""
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        {"relation": [["united states", "canada", "japan"], ["usa", "can", "jpn"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["united states", "canada", "germany"], ["usa", "can", "deu"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["japan", "germany", "france"], ["jpn", "deu", "fra"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["msft", "aapl"], ["microsoft", "apple"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["msft", "aapl", "googl"], ["microsoft", "apple", "alphabet"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["2024-01-01", "hello world", "83.5%", "the matrix", "blue"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    idx = tmp_path / "idx.pkl"
    result = subprocess.run(
        [sys.executable, "main.py",
         "--corpus_path", str(corpus_path),
         "--output_folder", str(out),
         "--threshold", "0.3",
         "--index_path", str(idx)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "filtered_corpus.jsonl").exists()
    assert (out / "coherence_distribution.png").exists()
    assert (out / "threshold_sweep.txt").exists()
    assert idx.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_wp1.py::test_main_end_to_end_on_synthetic_jsonl -v
```
Expected: failure (no `main.py` yet).

- [ ] **Step 3: Implement `main.py`**

Create `main.py`:

```python
"""WP1 pipeline driver.

Stages:
  1. Load corpus (JSONL or CSV folder)
  2. Build (or load cached) co-occurrence index
  3. Score every column for coherence
  4. Filter columns at threshold; emit JSONL + histogram + sweep table

Run:
    python main.py --corpus_path data/sample.json \\
                   --output_folder output/ \\
                   --threshold 0.3 \\
                   --index_path output/cooccurrence_index.pkl
"""

from __future__ import annotations

import argparse
import os
import time

from data_loader import load_corpus, corpus_summary
from cooccurrence_index import (
    build_cooccurrence_index,
    save_index,
    load_index,
    index_summary,
)
from npmi import score_corpus, test_npmi as npmi_sanity
from filter import (
    filter_corpus,
    rebuild_filtered_corpus,
    save_filtered_corpus,
    filtering_report,
    threshold_sweep,
    plot_coherence_distribution,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP1: PMI coherence filtering")
    p.add_argument("--corpus_path", required=True,
                   help="Path to JSONL file or folder of CSVs")
    p.add_argument("--output_folder", required=True,
                   help="Where to write filtered corpus and reports")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Coherence threshold (default 0.3)")
    p.add_argument("--index_path", default=None,
                   help="Path to save/load co-occurrence index "
                        "(default: <output_folder>/cooccurrence_index.pkl)")
    p.add_argument("--rebuild_index", action="store_true",
                   help="Force rebuilding the index even if cached")
    p.add_argument("--table_types", default="RELATION",
                   help="Comma-separated tableType values to keep (JSONL only). "
                        "Default: RELATION")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    index_path = args.index_path or os.path.join(
        args.output_folder, "cooccurrence_index.pkl"
    )
    table_types = tuple(t.strip() for t in args.table_types.split(",") if t.strip())

    total_start = time.time()

    # ---- Stage 1: Load ------------------------------------------------------
    print(f"[Stage 1/4] Loading corpus from {args.corpus_path}...")
    t0 = time.time()
    corpus = load_corpus(args.corpus_path, table_types=table_types)
    corpus_summary(corpus)
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 2: Index -----------------------------------------------------
    print(f"[Stage 2/4] Building co-occurrence index...")
    t0 = time.time()
    if (not args.rebuild_index) and os.path.exists(index_path):
        print(f"  Loading cached index from {index_path}")
        index = load_index(index_path)
    else:
        index = build_cooccurrence_index(corpus)
        save_index(index, index_path)
        print(f"  Saved index to {index_path}")
    index_summary(index)
    npmi_sanity(index)
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 3: Score -----------------------------------------------------
    print(f"[Stage 3/4] Computing coherence scores...")
    t0 = time.time()
    scored = score_corpus(corpus, index)
    if scored:
        avg = sum(s for _, _, _, s in scored) / len(scored)
        top = max(scored, key=lambda x: x[3])
        bot = min(scored, key=lambda x: x[3])
        print(f"  Scored {len(scored)} columns")
        print(f"  Average coherence score: {avg:.3f}")
        print(f"  Highest: {top[2][:5]}{'...' if len(top[2]) > 5 else ''} -> {top[3]:.3f}")
        print(f"  Lowest:  {bot[2][:5]}{'...' if len(bot[2]) > 5 else ''} -> {bot[3]:.3f}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 4: Filter ----------------------------------------------------
    print(f"[Stage 4/4] Filtering columns...")
    t0 = time.time()
    kept, removed = filter_corpus(scored, threshold=args.threshold)
    filtering_report(kept, removed)
    filtered = rebuild_filtered_corpus(corpus, kept)
    out_jsonl = os.path.join(args.output_folder, "filtered_corpus.jsonl")
    save_filtered_corpus(filtered, out_jsonl)
    print(f"  Saved filtered corpus to {out_jsonl}")
    plot_path = os.path.join(args.output_folder, "coherence_distribution.png")
    plot_coherence_distribution(scored, threshold=args.threshold, output_path=plot_path)
    print(f"  Saved histogram to {plot_path}")
    sweep_path = os.path.join(args.output_folder, "threshold_sweep.txt")
    threshold_sweep(scored, output_path=sweep_path)
    print(f"  Saved threshold sweep to {sweep_path}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    print(f"WP1 Complete! Filtered corpus ready for WP2.")
    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_wp1.py -v
```
Expected: all tests pass, including the end-to-end smoke test.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_wp1.py
git commit -m "main: CLI orchestration with stage timing and progress prints"
```

---

## Task 14: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Overwrite `README.md`**

```markdown
# Auto-Map — WP1: PMI Coherence Filtering

Re-implementation of Section 3.1 of Wang & He, *Synthesizing Mapping
Relationships Using Table Corpus* (SIGMOD 2017). Filters incoherent
columns out of a table corpus using mean pairwise NPMI.

## Install

```bash
pip install -r requirements.txt
```

## Run

JSONL input (web-table corpus format, e.g. `data/sample.json`):

```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/ \
               --threshold 0.3 \
               --index_path output/cooccurrence_index.pkl
```

CSV folder input:

```bash
python main.py --corpus_path path/to/csv_folder/ \
               --output_folder output/ \
               --threshold 0.3
```

Re-runs reuse the cached index from `--index_path` unless you pass
`--rebuild_index`.

## Outputs

| Path | What |
|---|---|
| `output/filtered_corpus.jsonl` | Filtered corpus, one table per line. Mirrors input schema with rejected columns removed; adds `coherence_scores` and `rejected_column_indices`. |
| `output/coherence_distribution.png` | Histogram of column coherence scores with a red dashed line at the threshold. |
| `output/threshold_sweep.txt` | Kept-vs-removed counts at thresholds {0.1, 0.2, 0.3, 0.4, 0.5}. |
| `output/cooccurrence_index.pkl` | Pickled `(cooccurrence, value_count, total_columns)` for re-runs. |

## How it works

1. **Load** the corpus (JSONL or CSV folder). For JSONL, only `RELATION`
   tables are kept and the header row is stripped per `hasHeader` /
   `headerRowIndex` metadata.
2. **Index** every distinct value and every distinct value pair across all
   columns of the corpus (one column = one observation; pairs are stored
   sorted so `(a,b)` and `(b,a)` collide).
3. **Score** each column as the mean NPMI over all unordered pairs of its
   distinct values.
4. **Filter** columns at the threshold, rebuild surviving tables (those
   with at least one kept column), and write outputs.

## Tests

```bash
python -m pytest -v
```

Tests run a deterministic synthetic mini-corpus that covers the math
(coherent country/iso and ticker/company columns score near +1; an
incoherent garbage column scores below the threshold).

## Spec and deviations

- Design spec: `docs/superpowers/specs/2026-05-05-wp1-pmi-coherence-design.md`
- Deviations from the original prompt: `claude/deviations.md`
- Source paper: `papers/automap.pdf` (Section 3.1)
```

- [ ] **Step 2: Verify README renders sanely**

```bash
head -40 README.md
```
Expected: top of the README starts with the title and an intro paragraph.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with install, run, outputs, and how-it-works"
```

---

## Task 15: End-to-end run on the real sample corpus

This is a manual verification task — no tests, just running the pipeline on `data/sample.json` and confirming sane outputs.

- [ ] **Step 1: Run the pipeline on `data/sample.json`**

```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/ \
               --threshold 0.3 \
               --index_path output/cooccurrence_index.pkl
```
Expected: four `[Stage k/4]` banners with timings, a filtering report, and a final `WP1 Complete!` line. No exceptions.

- [ ] **Step 2: Inspect the outputs**

```bash
ls -la output/
wc -l output/filtered_corpus.jsonl
cat output/threshold_sweep.txt
```
Expected:
- `filtered_corpus.jsonl` exists and has at least 1 line.
- `coherence_distribution.png` exists with non-zero size.
- `threshold_sweep.txt` shows monotonically decreasing kept counts as the threshold rises.

- [ ] **Step 3: Spot-check a few records**

```bash
head -3 output/filtered_corpus.jsonl | python -m json.tool --no-ensure-ascii 2>/dev/null || head -3 output/filtered_corpus.jsonl
```
Expected: each record has `relation`, `coherence_scores` (length matches `relation`), and `rejected_column_indices`.

- [ ] **Step 4: Open the histogram**

```bash
open output/coherence_distribution.png
```
Expected: a histogram that shows non-trivial mass on both sides of the threshold line — confirming the filter is doing real work, not trivially keeping or rejecting everything.

- [ ] **Step 5: If outputs look wrong, file a follow-up**

If anything is unexpected (all columns scoring near zero, threshold sweep flat, JSON parse errors), do NOT silently tweak — surface the symptom and stop. Likely root causes:
- Header rows weren't stripped (check `hasHeader` field on records).
- The corpus contains no `RELATION` tables (re-run with `--table_types RELATION,RELATION_OTHER` or similar after inspecting `data/sample.json`).
- A bug surfaced only at scale (a unit test should be added).

- [ ] **Step 6: Add `output/` to `.gitignore` and commit**

If `.gitignore` doesn't exist, create it. Append:

```
output/
__pycache__/
*.pyc
.pytest_cache/
```

Then:

```bash
git add .gitignore
git commit -m "chore: gitignore output and pyc artifacts"
```

---

## Self-review checklist

After implementation, before declaring done:

- [ ] Every spec section has at least one task above (matched: §2 inputs/outputs → Tasks 2,3,4,11; §3 math → Tasks 7,8; §4 architecture → all; §5 modules → 1:1 mapping; §6 schemas → Tasks 11,12; §7 testing → covered across all tasks; §9 deviations → respected by spec; §10 out of scope → none of the omitted features appear).
- [ ] No placeholders, no "TBD", no `# TODO` left in source.
- [ ] Every test in `tests/test_wp1.py` passes (`python -m pytest -v`).
- [ ] Function names match across tasks (`compute_pmi`, `compute_npmi`, `compute_coherence`, `score_corpus`, `filter_corpus`, `rebuild_filtered_corpus`, `save_filtered_corpus`, `filtering_report`, `threshold_sweep`, `plot_coherence_distribution`, `build_cooccurrence_index`, `save_index`, `load_index`, `index_summary`, `clean_value`, `_load_jsonl`, `_load_csv_folder`, `load_corpus`, `corpus_summary`, `test_npmi`).
- [ ] Type aliases consistent (`Table` in `data_loader`, `Index` in `cooccurrence_index`, `ScoredColumn` in `npmi`, `FilteredTable` in `filter`).
- [ ] CLI flags match the spec (`--corpus_path`, `--output_folder`, `--threshold`, `--index_path`, `--rebuild_index`, `--table_types`).
- [ ] Outputs land at the spec'd paths (`{output_folder}/filtered_corpus.jsonl`, `{output_folder}/coherence_distribution.png`, `{output_folder}/threshold_sweep.txt`, pickled index at `--index_path`).
- [ ] No `Co-Authored-By: Claude ...` trailers on any commit.
