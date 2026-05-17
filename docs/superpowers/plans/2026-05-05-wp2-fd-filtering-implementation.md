# WP2 FD Column-Pair Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Section 3.2 of Wang & He (SIGMOD 2017) — column-pair filtering by approximate FD — completing the paper's "Candidate Table Extraction" pipeline (WP1 + WP2). Includes a small WP1 row-alignment patch needed to make FD checking work.

**Architecture:** New `fd_filter.py` module with the math + IO; new `wp2.py` script for orchestration. Sequential to WP1: WP1 produces `output/filtered_corpus.jsonl`, WP2 reads that and produces `output/candidates.jsonl` with one ordered FD-passing column pair per line. Three-line patch to WP1 modules to keep `""` markers in row positions instead of dropping them per-column independently.

**Tech Stack:** Python 3.8+, stdlib only (collections, itertools, json, os, sys, argparse, time, typing), pytest for tests.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-05-wp2-fd-filtering-design.md`
- Source paper: `papers/automap.pdf` (Section 3.2 + Definition 2)
- WP1 deviations: `claude/deviations.md`
- WP1 implementation plan (already executed): `docs/superpowers/plans/2026-05-05-wp1-pmi-coherence-implementation.md`

**Important commit convention:** Do NOT include any `Co-Authored-By: Claude ...` trailer on commits. Plain commit messages only.

---

## File map

| Path | Status | Purpose |
|---|---|---|
| `data_loader.py` | patch | Preserve row alignment by keeping `""` markers in columns |
| `cooccurrence_index.py` | patch | Skip `""` when iterating distinct values |
| `npmi.py` | patch | Skip `""` in `compute_coherence` |
| `fd_filter.py` | create | `compute_approx_fd`, `filter_candidates_by_fd`, `save_candidates`, `candidates_summary` |
| `wp2.py` | create | argparse CLI; reads filtered_corpus.jsonl, writes candidates.jsonl |
| `conftest.py` | append | Add `fd_synthetic_table` fixture |
| `tests/test_wp1.py` | append | One new test for row-alignment preservation |
| `tests/test_wp2.py` | create | 12 tests covering compute_approx_fd, filter_candidates_by_fd, save, summary, end-to-end |
| `claude/deviations.md` | append | Add WP2 deviations section |
| `USAGE.md` | append | Add WP2 run/inspect sections |
| `WALKTHROUGH.md` | append | Add WP2 walkthrough section |

---

## Task 1: WP1 row-alignment patch

**Files:**
- Modify: `data_loader.py`
- Modify: `cooccurrence_index.py`
- Modify: `npmi.py`
- Modify: `tests/test_wp1.py`

The current loader drops empty values per column independently, which breaks row alignment between two columns of the same table. PMI doesn't notice (intra-column only) but FD will. This task is a single atomic commit that:
1. Keeps `""` markers in columns (preserves row alignment).
2. Skips `""` when iterating distinct values in the index and in coherence computation.
3. Adds one new test verifying alignment is preserved.

The patch is **behavior-preserving for PMI**: empty values were previously discarded by the loader before the index ever saw them; now they're discarded inside the index/coherence functions. Same effect, but columns now stay row-aligned for downstream consumers.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wp1.py` (place near other JSONL-loader tests, alongside `test_jsonl_loader_drops_empty_columns`):

```python
def test_jsonl_loader_preserves_row_alignment(tmp_path):
    """After the WP1 row-alignment patch, columns of the same table
    keep equal length even when some cells are empty. The "" markers
    are preserved in place; PMI/coherence skip them downstream."""
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["a", "", "c"], ["x", "y", "z"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    # Both columns kept; both have length 3 (alignment preserved).
    assert len(columns) == 2
    assert len(columns[0]) == 3
    assert len(columns[1]) == 3
    assert columns[0] == ["a", "", "c"]
    assert columns[1] == ["x", "y", "z"]
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp1.py::test_jsonl_loader_preserves_row_alignment -v
```
Expected: FAIL — current loader drops `""` from column 0 producing `["a", "c"]` of length 2.

- [ ] **Step 3: Patch `data_loader.py`**

In `_load_jsonl` (around lines 83-89), replace:

```python
            cleaned: List[List[str]] = []
            for col in relation:
                vals = [clean_value(v) for v in col]
                vals = [v for v in vals if v != ""]
                if len(set(vals)) < 2:
                    continue
                cleaned.append(vals)
```

with:

```python
            cleaned: List[List[str]] = []
            for col in relation:
                vals = [clean_value(v) for v in col]
                if len({v for v in vals if v != ""}) < 2:
                    continue
                cleaned.append(vals)
```

Apply the **identical** change in `_load_csv_folder` (around lines 124-130 — same five-line block).

- [ ] **Step 4: Patch `cooccurrence_index.py`**

In `build_cooccurrence_index`, replace:

```python
            distinct = sorted(set(col))
            if len(distinct) < 2:
                continue
```

with:

```python
            distinct = sorted(v for v in set(col) if v)
            if len(distinct) < 2:
                continue
```

- [ ] **Step 5: Patch `npmi.py`**

In `compute_coherence`, replace:

```python
    distinct = sorted(set(column))
    if len(distinct) < 2:
        return -1.0
```

with:

```python
    distinct = sorted(v for v in set(column) if v)
    if len(distinct) < 2:
        return -1.0
```

- [ ] **Step 6: Run the full test suite**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest -v
```
Expected: 51 passed (50 existing + 1 new). All previous tests continue to pass — the patch is behavior-preserving for PMI because empties were already excluded from the index path.

- [ ] **Step 7: Verify on real corpus (no commit)**

```bash
/Users/noah/coding/Automap/.venv/bin/python main.py --corpus_path data/sample.json \
    --output_folder output/ \
    --threshold 0.3 \
    --index_path output/cooccurrence_index.pkl \
    --rebuild_index
cat output/threshold_sweep.txt
```
Expected: numbers identical to pre-patch (~11611 kept / 345 removed at 0.3 — small drift of a few columns is acceptable, large drift means the patch changed semantics). Snapshot the threshold_sweep.txt counts in your head; they should match the WP1 final-run output:

```
Threshold | Kept | Removed | Kept %
----------+------+---------+-------
   0.1    | 11891 |    65   | 99.5%
   0.2    | 11811 |   145   | 98.8%
   0.3    | 11609 |   347   | 97.1%
   0.4    | 11019 |   937   | 92.2%
   0.5    | 10207 |  1749   | 85.4%
```

- [ ] **Step 8: Commit**

```bash
git add data_loader.py cooccurrence_index.py npmi.py tests/test_wp1.py
git commit -m "wp1: preserve row alignment in loader; index/coherence skip empties"
```

---

## Task 2: `compute_approx_fd` + fixture

**Files:**
- Create: `tests/test_wp2.py`
- Modify: `conftest.py`
- Create: `fd_filter.py`

This task adds the FD synthetic fixture, writes 8 tests covering all the math edge cases for `compute_approx_fd`, then implements the function. It's the most-tested unit of WP2 because every other piece composes on top.

- [ ] **Step 1: Add the `fd_synthetic_table` fixture to `conftest.py`**

Append to `conftest.py`:

```python
@pytest.fixture
def fd_synthetic_table():
    """Five row-aligned columns, 8 rows. Designed for FD test cases.

    Layout (one row per index, one column per list):
        col 0  LEFT_CC      col 1  RIGHT_CC    col 2  AMBIG
        col 3  FOREIGN      col 4  CONST

    Row 0:  united     usa       portland   yes  A
    Row 1:  united     usa       portland   yes  A
    Row 2:  canada     can       vancouver  no   A
    Row 3:  japan      jpn       tokyo      yes  A
    Row 4:  germany    deu       berlin     yes  A
    Row 5:  france     fra       paris      no   A
    Row 6:  portland   oregon    ""         yes  A     <- "portland" reused, ambiguous
    Row 7:  portland   maine     ""         no   A

    Designed so that:
      - (LEFT_CC, AMBIG) after empty-row filtering has 6 perfect-FD rows -> theta = 1.0
      - (LEFT_CC, RIGHT_CC) over the full 8 rows has theta = 6/8 = 0.75
      - (LEFT_CC, CONST) has only 1 distinct Y -> rejected
    """
    return ({}, [
        ["united", "united", "canada", "japan", "germany", "france", "portland", "portland"],
        ["usa",    "usa",    "can",    "jpn",   "deu",     "fra",    "oregon",   "maine"],
        ["portland","portland","vancouver","tokyo","berlin","paris","",          ""],
        ["yes",    "yes",    "no",     "yes",   "yes",     "no",     "yes",      "no"],
        ["A",      "A",      "A",      "A",     "A",       "A",      "A",        "A"],
    ])
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_wp2.py`:

```python
"""Unit tests for WP2 (column-pair filtering by approximate FD).

Tests grow incrementally per the implementation plan."""

from fd_filter import compute_approx_fd


def test_perfect_fd_passes(fd_synthetic_table):
    """LEFT_CC -> AMBIG has 2 empty rows (6, 7) which are dropped, leaving
    6 rows where each distinct LEFT value maps to exactly one AMBIG value.
    theta should be 1.0."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[2]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert theta == 1.0
    assert row_count == 6
    assert covered == 6


def test_name_ambiguity_rejected_at_095(fd_synthetic_table):
    """LEFT_CC -> RIGHT_CC over the full 8 rows: 'portland' appears twice
    mapping to two different right values. theta = 6/8 = 0.75 < 0.95."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[1]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert row_count == 8
    assert covered == 6
    assert theta == 0.75


def test_name_ambiguity_accepted_at_lower_theta(fd_synthetic_table):
    """The ambiguous case has theta = 0.75. With theta_threshold = 0.7
    the higher-level filter would accept it. compute_approx_fd itself
    doesn't apply the threshold — it just computes theta. Verifies the
    raw value is what we'd expect."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[1]
    theta, _, _, _ = compute_approx_fd(left, right)
    # Just confirms the same theta is reported regardless of caller threshold.
    assert theta == 0.75


def test_constant_column_rejected(fd_synthetic_table):
    """LEFT_CC -> CONST: Y has only 1 distinct value ('A'). Reject."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[4]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert theta == 0.0
    assert pairs == []
    assert row_count == 0


def test_too_few_rows_rejected():
    """A pair with fewer than min_rows non-empty rows is rejected."""
    left = ["a", "b", ""]
    right = ["x", "y", ""]
    theta, pairs, row_count, covered = compute_approx_fd(left, right, min_rows=3)
    assert theta == 0.0


def test_empty_rows_dropped(fd_synthetic_table):
    """LEFT_CC -> AMBIG: rows 6 and 7 have AMBIG=''; they're dropped before
    FD computation. Remaining 6 rows give a perfect FD."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[2]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert row_count == 6  # 8 total - 2 empty AMBIG rows


def test_pairs_are_deduplicated(fd_synthetic_table):
    """LEFT_CC -> AMBIG: 'united' appears twice mapping to 'portland' both
    times. The output 'pairs' list should contain ('united', 'portland')
    exactly once."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[2]
    _, pairs, _, _ = compute_approx_fd(left, right)
    united_entries = [p for p in pairs if p[0] == "united"]
    assert len(united_entries) == 1
    assert united_entries[0] == ("united", "portland")


def test_surviving_pairs_match_witness_subset(fd_synthetic_table):
    """For each x in pairs, the y is the most common y in the original
    column among rows where left == x and both values are non-empty."""
    from collections import Counter
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[1]  # has ambiguity
    _, pairs, _, _ = compute_approx_fd(left, right)
    # Build the ground-truth most-common-y for each x.
    rows = [(x, y) for x, y in zip(left, right) if x and y]
    by_x = {}
    for x, y in rows:
        by_x.setdefault(x, Counter())[y] += 1
    expected = {x: counter.most_common(1)[0][0] for x, counter in by_x.items()}
    actual = dict(pairs)
    assert actual == expected
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: ImportError on `fd_filter` (collection-level failure). All 8 tests blocked.

- [ ] **Step 4: Implement `compute_approx_fd`**

Create `fd_filter.py`:

```python
"""Approximate-FD filtering of column pairs (paper §3.2).

For each table from WP1, enumerate ordered column pairs (C_i, C_j),
i ≠ j. Apply X →_θ Y check per Definition 2 of the paper; keep pairs
with θ ≥ 0.95.

The witness subset R̄ is constructed greedily: for each distinct x in X,
pick the y that appears most often alongside it. This is the largest
such R̄ — picking any other y for a given x covers fewer rows.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

Candidate = Dict[str, Any]


def compute_approx_fd(
    left_col: List[str],
    right_col: List[str],
    *,
    min_rows: int = 3,
) -> Tuple[float, List[Tuple[str, str]], int, int]:
    """Approximate FD score X →_θ Y.

    Drops rows where left_col[k] == '' or right_col[k] == ''. For each
    distinct x, picks the most common y (witness-subset construction).

    Returns:
        (theta, surviving_pairs, row_count, covered_rows) where
          - theta = covered_rows / row_count
          - surviving_pairs is the deduped (x, most_common_y) list,
            one entry per distinct x
          - row_count = |R| (non-empty rows after row-pair filtering)
          - covered_rows = |R̄| (rows in the witness subset)
        Returns (0.0, [], 0, 0) if fewer than min_rows non-empty rows
        remain or either column has fewer than 2 distinct non-empty
        values.
    """
    rows = [(x, y) for x, y in zip(left_col, right_col) if x and y]
    if len(rows) < min_rows:
        return 0.0, [], 0, 0
    by_x: Dict[str, Counter] = defaultdict(Counter)
    for x, y in rows:
        by_x[x][y] += 1
    distinct_y = {y for _, y in rows}
    if len(by_x) < 2 or len(distinct_y) < 2:
        return 0.0, [], 0, 0
    covered = 0
    surviving_pairs: List[Tuple[str, str]] = []
    for x, counter in by_x.items():
        y, cnt = counter.most_common(1)[0]
        covered += cnt
        surviving_pairs.append((x, y))
    return covered / len(rows), surviving_pairs, len(rows), covered
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add conftest.py fd_filter.py tests/test_wp2.py
git commit -m "fd_filter: compute_approx_fd with witness-subset construction"
```

---

## Task 3: `filter_candidates_by_fd`

**Files:**
- Modify: `fd_filter.py`
- Modify: `tests/test_wp2.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wp2.py`:

```python
from fd_filter import filter_candidates_by_fd


def test_ordered_pairs_are_distinct_candidates(fd_synthetic_table):
    """A 2-column table evaluated with FD yields up to 2 candidates:
    (left=0, right=1) and (left=1, right=0). They're independent and
    can have different theta values."""
    _, cols = fd_synthetic_table
    # Use just LEFT_CC and AMBIG as a 2-column table (perfect FD both
    # ways after empty-row filtering on either side).
    record = {
        "relation": [cols[0], cols[2]],
        "coherence_scores": [1.0, 1.0],
        "rejected_column_indices": [],
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    indices = {(c["left_column_index"], c["right_column_index"]) for c in candidates}
    assert (0, 1) in indices
    assert (1, 0) in indices


def test_filter_candidates_passes_through_metadata():
    """source_metadata should pass through input record fields except
    relation, coherence_scores, and rejected_column_indices."""
    record = {
        "relation": [
            ["a", "b", "c", "d"],
            ["1", "2", "3", "4"],
        ],
        "coherence_scores": [1.0, 1.0],
        "rejected_column_indices": [2, 5],
        "pageTitle": "Hello",
        "url": "http://example.com",
        "tableType": "RELATION",
        "tableNum": 7,
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    assert len(candidates) >= 1
    meta = candidates[0]["source_metadata"]
    assert meta["pageTitle"] == "Hello"
    assert meta["url"] == "http://example.com"
    assert meta["tableType"] == "RELATION"
    assert meta["tableNum"] == 7
    assert "relation" not in meta
    assert "coherence_scores" not in meta
    assert "rejected_column_indices" not in meta


def test_filter_candidates_includes_full_schema():
    """Every produced candidate has the 8 spec'd fields with correct types."""
    record = {
        "relation": [
            ["a", "b", "c", "d"],
            ["1", "2", "3", "4"],
        ],
        "pageTitle": "X",
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    assert len(candidates) >= 1
    c = candidates[0]
    assert isinstance(c["pairs"], list)
    assert isinstance(c["theta"], float)
    assert isinstance(c["row_count"], int)
    assert isinstance(c["covered_rows"], int)
    assert isinstance(c["source_table_index"], int)
    assert isinstance(c["left_column_index"], int)
    assert isinstance(c["right_column_index"], int)
    assert isinstance(c["source_metadata"], dict)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 3 new failures referencing `filter_candidates_by_fd`.

- [ ] **Step 3: Implement `filter_candidates_by_fd`**

Append to `fd_filter.py`:

```python
def filter_candidates_by_fd(
    filtered_records: Iterable[Dict[str, Any]],
    *,
    theta_threshold: float = 0.95,
    min_rows: int = 3,
) -> List[Candidate]:
    """For each filtered table record, enumerate ordered column pairs
    (i, j) with i ≠ j. Run compute_approx_fd; keep pairs with
    θ ≥ theta_threshold. Returns one Candidate per surviving pair.

    A Candidate is a dict with the schema documented in the WP2 design
    spec (`pairs`, `theta`, `row_count`, `covered_rows`,
    `source_table_index`, `left_column_index`, `right_column_index`,
    `source_metadata`).
    """
    candidates: List[Candidate] = []
    excluded_metadata_keys = {"relation", "coherence_scores", "rejected_column_indices"}
    for table_idx, record in enumerate(filtered_records):
        relation = record.get("relation", [])
        n = len(relation)
        if n < 2:
            continue
        source_metadata = {
            k: v for k, v in record.items() if k not in excluded_metadata_keys
        }
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                theta, pairs, row_count, covered = compute_approx_fd(
                    relation[i], relation[j], min_rows=min_rows
                )
                if theta >= theta_threshold:
                    candidates.append({
                        "pairs": [list(p) for p in pairs],
                        "theta": theta,
                        "row_count": row_count,
                        "covered_rows": covered,
                        "source_table_index": table_idx,
                        "left_column_index": i,
                        "right_column_index": j,
                        "source_metadata": source_metadata,
                    })
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 11 passed (8 + 3).

- [ ] **Step 5: Commit**

```bash
git add fd_filter.py tests/test_wp2.py
git commit -m "fd_filter: filter_candidates_by_fd with metadata pass-through"
```

---

## Task 4: `save_candidates`

**Files:**
- Modify: `fd_filter.py`
- Modify: `tests/test_wp2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wp2.py`:

```python
import json
from fd_filter import save_candidates


def test_save_candidates_jsonl_schema(fd_synthetic_table, tmp_path):
    """Round-trip: save_candidates writes valid JSONL where each line
    parses back to the same Candidate dict."""
    _, cols = fd_synthetic_table
    record = {
        "relation": [cols[0], cols[2]],
        "pageTitle": "Test",
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    out = tmp_path / "candidates.jsonl"
    save_candidates(candidates, str(out))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(candidates)
    for line, original in zip(lines, candidates):
        parsed = json.loads(line)
        # 8 required fields
        for field in ("pairs", "theta", "row_count", "covered_rows",
                      "source_table_index", "left_column_index",
                      "right_column_index", "source_metadata"):
            assert field in parsed
        # JSON converts tuples to lists; original `pairs` was already lists.
        assert parsed["theta"] == original["theta"]
        assert parsed["pairs"] == original["pairs"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py::test_save_candidates_jsonl_schema -v
```
Expected: ImportError on `save_candidates`.

- [ ] **Step 3: Implement `save_candidates`**

Append to `fd_filter.py`:

```python
def save_candidates(candidates: List[Candidate], output_path: str) -> None:
    """Write JSONL, one candidate per line. Creates parent directory
    if missing."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add fd_filter.py tests/test_wp2.py
git commit -m "fd_filter: save_candidates writes JSONL with full schema"
```

---

## Task 5: `candidates_summary`

**Files:**
- Modify: `fd_filter.py`
- Modify: `tests/test_wp2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wp2.py`:

```python
from fd_filter import candidates_summary


def test_candidates_summary_runs(fd_synthetic_table, capsys):
    _, cols = fd_synthetic_table
    record = {
        "relation": [cols[0], cols[2]],
        "pageTitle": "Test",
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    candidates_summary(candidates)
    out = capsys.readouterr().out
    assert "Candidates" in out
    assert "Theta" in out or "theta" in out


def test_candidates_summary_handles_empty_input(capsys):
    candidates_summary([])
    out = capsys.readouterr().out
    assert "No candidates" in out or "0 candidates" in out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 2 new failures referencing `candidates_summary`.

- [ ] **Step 3: Implement `candidates_summary`**

Append to `fd_filter.py`:

```python
def candidates_summary(candidates: List[Candidate]) -> None:
    """Print summary stats for a candidate list to stdout: count,
    distinct source tables, theta distribution (mean/min/max), and
    the top-5 candidates by theta."""
    if not candidates:
        print("  No candidates produced")
        return
    n = len(candidates)
    n_tables = len({c["source_table_index"] for c in candidates})
    thetas = [c["theta"] for c in candidates]
    mean_t = sum(thetas) / n
    print(f"  Candidates: {n}")
    print(f"  Source tables represented: {n_tables}")
    print(f"  Theta: mean={mean_t:.3f}, min={min(thetas):.3f}, max={max(thetas):.3f}")
    top = sorted(candidates, key=lambda c: c["theta"], reverse=True)[:5]
    print(f"  Top 5 by theta:")
    for c in top:
        sample = c["pairs"][:3]
        more = "..." if len(c["pairs"]) > 3 else ""
        print(f"    theta={c['theta']:.3f} rows={c['row_count']} pairs={sample}{more}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add fd_filter.py tests/test_wp2.py
git commit -m "fd_filter: candidates_summary with theta distribution and top-5"
```

---

## Task 6: `wp2.py` CLI + end-to-end smoke test

**Files:**
- Create: `wp2.py`
- Modify: `tests/test_wp2.py`

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/test_wp2.py`:

```python
import subprocess
import sys


def test_wp2_end_to_end_smoke(tmp_path):
    """Smoke test: write a tiny filtered_corpus.jsonl, run wp2.py against
    it via subprocess, verify candidates.jsonl is produced and parses."""
    filtered = tmp_path / "filtered.jsonl"
    records = [
        {
            "relation": [
                ["united", "united", "canada", "japan", "germany", "france"],
                ["usa",    "usa",    "can",    "jpn",   "deu",     "fra"],
            ],
            "coherence_scores": [1.0, 1.0],
            "rejected_column_indices": [],
            "pageTitle": "Country test",
            "tableType": "RELATION",
        },
    ]
    with open(filtered, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "wp2.py",
         "--filtered_corpus", str(filtered),
         "--output_folder", str(out_dir),
         "--theta", "0.95"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out_path = out_dir / "candidates.jsonl"
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # both ordered pairs (0,1) and (1,0) survive
    for line in lines:
        parsed = json.loads(line)
        assert "theta" in parsed
        assert parsed["theta"] >= 0.95
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py::test_wp2_end_to_end_smoke -v
```
Expected: FAIL — `wp2.py` doesn't exist yet, subprocess returncode non-zero.

- [ ] **Step 3: Implement `wp2.py`**

Create `wp2.py`:

```python
"""WP2 pipeline driver — column-pair filtering by approximate FD.

Stages:
  1. Load filtered corpus (WP1 output JSONL).
  2. For each table, enumerate ordered column pairs and apply approximate
     FD checking. Keep pairs with theta >= --theta. Write candidates.jsonl.

Run:
    python wp2.py --filtered_corpus output/filtered_corpus.jsonl \\
                  --output_folder output/ \\
                  --theta 0.95
"""

from __future__ import annotations

import argparse
import json
import os
import time

from fd_filter import (
    filter_candidates_by_fd,
    save_candidates,
    candidates_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WP2: column-pair filtering by approximate FD"
    )
    p.add_argument("--filtered_corpus", default="output/filtered_corpus.jsonl",
                   help="Path to WP1 output JSONL "
                        "(default: output/filtered_corpus.jsonl)")
    p.add_argument("--output_folder", default="output/",
                   help="Where to write candidates (default: output/)")
    p.add_argument("--theta", type=float, default=0.95,
                   help="Approximate-FD threshold (default 0.95)")
    p.add_argument("--min_rows", type=int, default=3,
                   help="Minimum non-empty rows for a pair to be evaluated "
                        "(default 3)")
    p.add_argument("--output_filename", default="candidates.jsonl",
                   help="Output filename (default candidates.jsonl)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, args.output_filename)

    total_start = time.time()

    # ---- Stage 1: Load --------------------------------------------------
    print(f"[Stage 1/2] Loading filtered corpus from {args.filtered_corpus}...")
    t0 = time.time()
    records = []
    with open(args.filtered_corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} filtered tables")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 2: FD filter ---------------------------------------------
    print(f"[Stage 2/2] FD filtering (theta={args.theta}, "
          f"min_rows={args.min_rows})...")
    t0 = time.time()
    candidates = filter_candidates_by_fd(
        records, theta_threshold=args.theta, min_rows=args.min_rows
    )
    candidates_summary(candidates)
    save_candidates(candidates, out_path)
    print(f"  Saved {len(candidates)} candidates to {out_path}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    print(f"WP2 Complete!")
    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/Users/noah/coding/Automap/.venv/bin/python -m pytest tests/test_wp2.py -v
```
Expected: 15 passed (14 + 1 smoke).

- [ ] **Step 5: Commit**

```bash
git add wp2.py tests/test_wp2.py
git commit -m "wp2: CLI driver with stage timing and end-to-end smoke test"
```

---

## Task 7: End-to-end run on real corpus

This is a manual verification task — no tests, just running WP2 against the real `output/filtered_corpus.jsonl` from WP1 and confirming sensible output.

- [ ] **Step 1: Re-run WP1 to refresh the filtered corpus**

(The row-alignment patch from Task 1 means the index pickle may produce slightly different numerics; safest to rebuild.)

```bash
/Users/noah/coding/Automap/.venv/bin/python main.py \
    --corpus_path data/sample.json \
    --output_folder output/ \
    --threshold 0.3 \
    --index_path output/cooccurrence_index.pkl \
    --rebuild_index
```
Expected: `WP1 Complete!` with the 4 stage banners. Confirm `output/filtered_corpus.jsonl` exists with ~2700 lines.

- [ ] **Step 2: Run WP2**

```bash
/Users/noah/coding/Automap/.venv/bin/python wp2.py \
    --filtered_corpus output/filtered_corpus.jsonl \
    --output_folder output/ \
    --theta 0.95
```
Expected: 2 stage banners, candidates summary printed, `output/candidates.jsonl` produced. No exceptions.

- [ ] **Step 3: Inspect the output**

```bash
ls -la output/candidates.jsonl
wc -l output/candidates.jsonl
head -1 output/candidates.jsonl | /Users/noah/coding/Automap/.venv/bin/python -m json.tool
```
Expected: file exists, has at least a few hundred lines, the first line parses as JSON with all 8 required fields.

- [ ] **Step 4: Sanity check the candidate distribution**

```bash
/Users/noah/coding/Automap/.venv/bin/python -c "
import json
candidates = []
with open('output/candidates.jsonl') as f:
    for line in f:
        candidates.append(json.loads(line))
print(f'Total candidates: {len(candidates)}')
print(f'Distinct source tables: {len({c[\"source_table_index\"] for c in candidates})}')
thetas = [c['theta'] for c in candidates]
print(f'Theta: mean={sum(thetas)/len(thetas):.3f}, min={min(thetas):.3f}, max={max(thetas):.3f}')
print(f'All theta >= 0.95: {all(t >= 0.95 for t in thetas)}')
print(f'Top 3 by theta:')
for c in sorted(candidates, key=lambda c: c['theta'], reverse=True)[:3]:
    print(f'  theta={c[\"theta\"]:.3f} rows={c[\"row_count\"]} pairs[:3]={c[\"pairs\"][:3]}')
"
```
Expected: every theta >= 0.95 (filter sanity); a few hundred to a few thousand candidates total; top candidates look like real mappings (numbers / codes / names).

- [ ] **Step 5: If outputs look anomalous, surface it**

If theta floor is below 0.95, or candidates count is 0, or all candidates look like garbage, do NOT silently tweak. Surface the symptom. Likely root causes:
- Filter logic bug — re-check `filter_candidates_by_fd` against the spec.
- Real-corpus tables have unusual structure that the synthetic fixture doesn't cover.

- [ ] **Step 6: Commit (no code change, just an output artifact note)**

If a `.gitignore` already covers `output/`, no commit is needed — the data files don't enter git. If you want to commit a snapshot of stats for the record:

(skip — the existing `.gitignore` covers `output/`)

---

## Task 8: Update `claude/deviations.md`

**Files:**
- Modify: `claude/deviations.md`

- [ ] **Step 1: Append the WP2 deviations section**

Open `claude/deviations.md`. Find the end of the existing summary table (the last row should be the WP1 deviation #13 about the synthetic corpus expansion). After the existing "What is NOT a deviation" section, append:

```markdown

---

## WP2 — Section 3.2 (Column-Pair Filtering by FD)

The original prompt (`wp1_prompt.md`) only covered Section 3.1. WP2 was
designed against the paper directly. Notable choices:

| # | Choice | Rationale |
|---|---|---|
| 14 | **JSONL output with deduplicated `(l, r)` pairs** | Matches paper §4.1 data model `B = {(l_i, r_i)}` directly. WP3 will need this set form for compatibility scoring. |
| 15 | **WP1 row-alignment patch** (`""` markers preserved in columns; PMI/coherence skip them) | The original loader dropped empties per column independently, breaking row alignment. PMI didn't notice (intra-column only) but FD requires aligned columns. WP1 filter decisions are unchanged. |
| 16 | **`min_rows = 3` for FD eligibility** | Below 3 non-empty rows, FD score is meaningless. Configurable via `--min_rows`. |
| 17 | **Reject pairs with <2 distinct values on either side** | Constant columns (Y always the same) trivially "satisfy" FD but carry no mapping signal. Same spirit as WP1's "<2 unique" column rule. |
| 18 | **Greedy witness-subset construction** | For each distinct x, pick the most-common y. Provably the largest R̄ — picking any other y for a given x covers fewer rows. |
| 19 | **No reverse-direction inference** | (C_i, C_j) and (C_j, C_i) evaluated independently. A 1:1 mapping survives both directions; an N:1 mapping survives one. Matches the paper's distinction between 1:1 and N:1 (§2.1). |
```

- [ ] **Step 2: Verify the edit**

```bash
tail -30 claude/deviations.md
```
Expected: the new WP2 section is present and well-formed.

- [ ] **Step 3: Commit**

```bash
git add claude/deviations.md
git commit -m "docs: deviations log for WP2 design choices"
```

---

## Task 9: Update `USAGE.md`

**Files:**
- Modify: `USAGE.md`

- [ ] **Step 1: Append the WP2 sections**

The current `USAGE.md` has 7 numbered sections (Setup, Running, CLI flags, Input, Output, Interpreting scores, Tests, Troubleshooting). Add a new section after "Outputs" specifically for WP2 — both running and interpreting candidates.

Open `USAGE.md` and find the end of section "## 4. Output files" (ends just before "## 5. Interpreting coherence scores"). Insert the following BETWEEN section 4 and section 5, then renumber subsequent sections:

```markdown
---

## 5. Running WP2 (column-pair FD filtering)

Once WP1 has produced `output/filtered_corpus.jsonl`, run WP2 to enumerate
ordered column pairs and keep those that satisfy approximate FD:

```bash
python wp2.py --filtered_corpus output/filtered_corpus.jsonl \
              --output_folder output/ \
              --theta 0.95
```

You'll see two stage banners:

```
[Stage 1/2] Loading filtered corpus from output/filtered_corpus.jsonl...
  Loaded NNNN filtered tables
  Time: ...

[Stage 2/2] FD filtering (theta=0.95, min_rows=3)...
  Candidates: NNN
  Source tables represented: NNN
  Theta: mean=..., min=..., max=...
  Top 5 by theta:
    theta=... rows=... pairs=[...]
    ...
  Saved NNN candidates to output/candidates.jsonl
  Time: ...

WP2 Complete!
```

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--filtered_corpus` | `output/filtered_corpus.jsonl` | Path to WP1 output JSONL |
| `--output_folder` | `output/` | Where to write `candidates.jsonl` |
| `--theta` | `0.95` | Approximate-FD threshold (paper-mandated). Pairs with θ < this are rejected. |
| `--min_rows` | `3` | Minimum non-empty rows for a pair to be evaluated. Below this, FD has no meaning. |
| `--output_filename` | `candidates.jsonl` | Output filename within `--output_folder` |

WP2 is fast (no global index) — iterating on θ is cheap. Re-run with a
different `--theta` to widen or narrow the candidate set.

---

## 6. Interpreting `candidates.jsonl`

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

Re-run `wp2.py` at multiple thresholds to compare candidate counts; this
is much cheaper than re-running WP1.

---
```

After inserting, the section numbers below should be renumbered: section 5 (was "Interpreting coherence scores") becomes section 7, section 6 ("Running tests") becomes section 8, section 7 ("Troubleshooting") becomes section 9.

Update the table of contents / heading levels accordingly. Specifically: search for `## 5. Interpreting coherence scores` and rename to `## 7. Interpreting coherence scores`. `## 6. Running tests` → `## 8. Running tests`. `## 7. Troubleshooting` → `## 9. Troubleshooting`.

Also update the "How it works" mention (in the original USAGE.md, the run-command says `python main.py ...`) — find that and add a note that running `wp2.py` after `main.py` completes Section 3 of the paper.

- [ ] **Step 2: Verify the edit**

```bash
grep -n "^## " USAGE.md
```
Expected: numbered sections 1-9 in order.

- [ ] **Step 3: Commit**

```bash
git add USAGE.md
git commit -m "docs: USAGE adds WP2 run command and candidates.jsonl interpretation"
```

---

## Task 10: Update `WALKTHROUGH.md`

**Files:**
- Modify: `WALKTHROUGH.md`

- [ ] **Step 1: Append the WP2 walkthrough section**

The current `WALKTHROUGH.md` has 10 numbered sections, ending with section 10 ("If someone asks you a question"). Add a new top-level section 8.5 (or section 11 — extending the linear flow) for WP2, plus update existing sections that mention Section 3 scope.

Specifically:

**(a)** In section 1 ("The problem we're solving"), update the sub-step list to mark WP2 done:

Find:
```
- **Step 3.1: Column filtering by PMI** ← we're here
- **Step 3.2: Column-pair filtering by FD** (functional dependency) — that's WP2
```

Replace with:
```
- **Step 3.1: Column filtering by PMI** — WP1 (done)
- **Step 3.2: Column-pair filtering by FD** — WP2 (done) ← we now cover both
```

**(b)** In section 8 ("What's NOT in WP1"), rename to "What's NOT in WP1+WP2" and remove the FD-filtering bullet:

Find the existing section 8 (the "What's NOT in WP1" header). Replace its title and intro:

```
## 8. What's NOT in WP1+WP2

- **Table synthesis** (paper §4) — WP3.
- **Conflict resolution** (paper §5) — WP4.
- **Streaming / out-of-core** — corpus is loaded into memory in one pass.
  Fine for sample-scale, would need rework for the paper's 100M-table
  experiments.
- **Distributed execution** — paper uses MapReduce. We run in-process.
- **Bidirectional FD inference** — `(C_i, C_j)` and `(C_j, C_i)` are
  evaluated as independent candidates. WP3 can later detect 1:1 mappings
  by finding both directions present in the candidate set.
```

(Remove the "Functional-dependency filtering" bullet that was previously listed as out-of-scope.)

**(c)** Add a new section after section 5 ("The five modules") and before section 6 ("The synthetic test corpus"). Renumber subsequent sections accordingly (synthetic corpus becomes 7, key design decisions becomes 8, etc.).

The new section:

```markdown
---

## 6. WP2: Column-pair filtering by FD

WP1 throws away **incoherent columns**. WP2 throws away **incoherent
column pairs** — pairs that survive PMI but don't actually express a
mapping relationship.

### The problem

After WP1, each surviving table has columns that all individually carry
semantic signal. But two coherent columns of the same table aren't
automatically a mapping pair. The paper's example: a table with `Home
Team`, `Away Team`, `Date`, `Stadium`, `Location` columns. Each is a
coherent column on its own (PMI passes). But `(Home Team, Away Team)`
isn't a mapping — both teams change game by game; one doesn't determine
the other. Only `(Home Team, Stadium)` and `(Stadium, Home Team)` are
real mappings.

So we need a per-pair filter on top of the per-column one. That filter
is approximate **functional dependency**.

### The math (paper Definition 2)

For two columns X and Y of the same table, with rows aligned, we ask:

> Is there a subset R̄ of the rows, with |R̄| ≥ 0.95 |R|, where every
> distinct x value in X maps to exactly one y value in Y?

The largest such subset is built greedily: for each distinct x in X,
pick the y value that appears most often alongside it. Every row where
x maps to its most-frequent y is in R̄; rows where x maps to a different
y are excluded.

```
For each distinct x in X:
    most_common_y(x) = mode of y values that co-occur with x
    covered += count of rows where (X = x AND Y = most_common_y(x))

θ = covered / |R|
```

This greedy choice is provably optimal — picking any other y for a given
x would only cover fewer rows.

### The Portland example (why approximate, not strict)

The paper uses Portland → Oregon vs Portland → Maine to motivate
approximate FD. Suppose the rows are:

```
(usa, dollar)        (usa → dollar perfectly)
(usa, dollar)
(usa, dollar)
(canada, cad)        (canada → cad perfectly)
(canada, cad)
(portland, oregon)   <- ambiguous
(portland, maine)    <- ambiguous
```

For `usa`: most_common = `dollar`, count 3.
For `canada`: most_common = `cad`, count 2.
For `portland`: most_common = either, count 1 (tie).

|R| = 7, covered = 3+2+1 = 6, θ ≈ 0.857. **Below 0.95 → rejected.**

If we lowered the threshold to 0.85 the pair would survive — that's how
the user accepts more name-ambiguous mappings at the cost of more
spurious ones.

### Why ordered pairs

`(C_i, C_j)` and `(C_j, C_i)` are evaluated independently. A **1:1
mapping** (every x ↔ exactly one y) survives both directions. An
**N:1 mapping** (many x's share one y) survives only X→Y. The paper
distinguishes 1:1 vs N:1 in §2.1 and uses the directionality downstream
in WP3 — so we keep both directions in the output rather than collapsing.

### The pipeline extension

```
WP1 (main.py)
   │
   ▼
output/filtered_corpus.jsonl     <-- one table per line, row-aligned columns
   │
   ▼
WP2 (wp2.py)
   for each table:
     for each ordered (i, j), i ≠ j:
       drop empty rows pairwise
       compute approximate FD via witness-subset construction
       if θ ≥ 0.95: emit candidate
   │
   ▼
output/candidates.jsonl          <-- one ordered column pair per line
```

The new module is `fd_filter.py` (math + IO) and the orchestration script
is `wp2.py`. Only one small change needed in WP1: the loader now keeps
`""` markers in columns to preserve row alignment (which PMI didn't
require but FD does). PMI/coherence functions skip `""` when iterating
distinct values, so the WP1 outputs are unchanged.

### Output schema

Each candidate is a JSON object on its own line, with eight fields:

| Field | Meaning |
|---|---|
| `pairs` | Deduplicated `(left, right)` value pairs from R̄. One entry per distinct left. |
| `theta` | The approximate-FD score, in `[0.95, 1.0]`. |
| `row_count` | `|R|` — non-empty rows used. |
| `covered_rows` | `|R̄|` — rows in the witness subset. |
| `source_table_index` | Line number in `filtered_corpus.jsonl`. |
| `left_column_index` | Index of left column in the filtered table's `relation`. |
| `right_column_index` | Index of right column. |
| `source_metadata` | Pass-through metadata (page title, URL, etc.). |

WP3 will treat each candidate's `pairs` as a set `B = {(l_i, r_i)}`
exactly as defined in paper §4.1, and find candidates B, B' that are
compatible (large overlap → same relationship → merge).

```

**(d)** In section 10 ("If someone asks you a question"), append three more Q&A entries:

```markdown
> **"What's the difference between WP1's PMI filtering and WP2's FD filtering?"**
> WP1 looks at *individual columns* and asks "do these values belong
> together semantically?" using corpus-wide co-occurrence statistics.
> WP2 looks at *pairs of columns* in the same table and asks "does the
> first column functionally determine the second?" using local row-level
> statistics. PMI removes garbage columns; FD removes column pairs that
> aren't real mappings even if both columns are individually coherent.

> **"Why approximate FD instead of strict?"**
> Strict FD would reject every table with name ambiguity (Portland →
> Oregon and Portland → Maine cause strict FD to fail). The paper allows
> 5% noise via the θ threshold, which is enough to admit real-world
> mappings while still rejecting clearly non-functional pairs like (Home
> Team, Away Team).

> **"Why is the threshold 0.95 specifically?"**
> Paper §2.1 says they consider θ over 95%. It's an empirical choice that
> balances false negatives (rejecting real mappings due to occasional
> ambiguity) against false positives (accepting non-mappings that happen
> to be mostly-functional in one table). Like the WP1 threshold, it's
> tunable and downstream stages validate the choice.
```

- [ ] **Step 2: Verify the edit**

```bash
grep -n "^## " WALKTHROUGH.md
```
Expected: section list shows the new "WP2: Column-pair filtering by FD" section, and section 8 is renamed.

- [ ] **Step 3: Commit**

```bash
git add WALKTHROUGH.md
git commit -m "docs: WALKTHROUGH adds WP2 section with FD math and Portland example"
```

---

## Self-review checklist

After implementation, before declaring done:

- [ ] Every spec section has at least one task (matched: §1 Goal → all tasks; §2 I/O → Tasks 6, 7; §3 Math → Task 2; §4 WP1 patch → Task 1; §5 Architecture → Task 6; §6 Module specs → Tasks 2-5; §7 Output schema → Tasks 4, 6; §8 Testing → Tasks 1-6; §9 Doc updates → Tasks 8-10; §10 Constraints → all; §11 Out of scope → none of the omitted features appear; §12 Deliverables → all 11 paths in the file map)
- [ ] No placeholders, no "TBD", no `# TODO` left in source.
- [ ] All 51 WP1 tests still pass after Task 1 (50 existing + 1 new alignment test).
- [ ] All 15 WP2 tests pass after Task 6 (12 from spec + 3 defensive: full-schema check, summary smoke, summary empty-input).
- [ ] Total: 66 passing tests.
- [ ] Function names match across tasks (`compute_approx_fd`, `filter_candidates_by_fd`, `save_candidates`, `candidates_summary`).
- [ ] Type aliases consistent (`Candidate` in `fd_filter.py`).
- [ ] CLI flags match the spec (`--filtered_corpus`, `--output_folder`, `--theta`, `--min_rows`, `--output_filename`).
- [ ] Output lands at the spec'd path (`{output_folder}/candidates.jsonl`).
- [ ] WP1's PMI numbers on real corpus are unchanged after Task 1's patch (within rounding).
- [ ] WP2 on real corpus produces a non-trivial candidate set (hundreds-thousands).
- [ ] No `Co-Authored-By: Claude ...` trailers on any commit.
