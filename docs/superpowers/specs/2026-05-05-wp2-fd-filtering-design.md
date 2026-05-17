# WP2 — Column-Pair Filtering by FD: Design Spec

**Date:** 2026-05-05
**Project:** Auto-Map (re-implementation of Wang & He, SIGMOD 2017, *Synthesizing Mapping Relationships Using Table Corpus*)
**Scope:** Section 3.2 of the paper — Column-Pair Filtering by approximate Functional Dependency. With WP1 already complete, WP2 finishes Section 3 (Candidate Table Extraction). WP3 (Section 4 Table Synthesis) and WP4 (Section 5 Conflict Resolution) are out of scope.

---

## 1. Goal

Given the WP1 output (filtered corpus where each surviving column has coherence ≥ τ), enumerate ordered column pairs `(C_i, C_j)` from each table and emit those that satisfy approximate functional dependency `X →_θ Y` with θ ≥ 0.95. The result is the candidate set ready for WP3.

## 2. Inputs and outputs

### Input
- **Primary:** `output/filtered_corpus.jsonl` (WP1 output). Each line is a filtered table with row-aligned `relation`, plus `coherence_scores`, `rejected_column_indices`, and pass-through metadata.

### Output
- `output/candidates.jsonl` — one FD-passing ordered pair per line.

## 3. Math (paper §3.2 + Definition 2)

**Definition 2 (paper §2.1):**
> Let R be a conceptual relation with two attributes X, Y. The relationship is a θ-approximate mapping relationship M_θ(X, Y), denoted X →_θ Y, if there exists a subset R̄ ⊆ R with |R̄| ≥ θ|R|, in which all x ∈ X functionally determines exactly one y ∈ Y.

**Witness-subset construction.** For each distinct x in X, the witness subset R̄ contains every row where x maps to its most-frequent y. This is provably the largest such R̄ — picking any other y for a given x covers fewer rows.

```
For each distinct x in X(R):
    most_common_y(x) = mode of y values that co-occur with x
    covered += count of rows where (X = x AND Y = most_common_y(x))

θ = covered / |R|
```

A pair `(C_i, C_j)` is **kept** iff `θ ≥ 0.95`. Default θ = 0.95.

### Worked example

For rows `[(usa,dollar), (usa,dollar), (usa,dollar), (canada,cad), (canada,cad), (portland,oregon), (portland,maine)]`:

- `usa` → `dollar` (count 3)
- `canada` → `cad` (count 2)
- `portland` → `oregon` *or* `maine` (count 1 either way; tie broken arbitrarily)

|R| = 7, covered = 3 + 2 + 1 = 6, θ ≈ 0.857. Below 0.95 — pair rejected. This is exactly the "Portland → Oregon vs Portland → Maine" name-ambiguity case from §3.2.

### Edge cases

| Case | Behavior |
|---|---|
| <3 non-empty rows after row-pair filtering | θ = 0, reject (`min_rows=3`) |
| X has <2 distinct non-empty values | θ = 0, reject (no real signal) |
| Y has <2 distinct non-empty values | θ = 0, reject (constant column) |
| Tie in most-common-y for some x | Tiebreak by `Counter.most_common(1)` (first encountered). Affects `surviving_pairs` content, not θ. |
| All x distinct (each appears once) | θ = 1.0. Kept. This is correct — a column where every value is unique is a strict-FD mapping. |

## 4. WP1 row-alignment patch

WP1's `data_loader.py` currently drops empty cells per column independently:

```python
vals = [v for v in vals if v != ""]
```

This breaks row alignment between columns of the same table. PMI doesn't notice (intra-column only) but FD requires aligned columns. Patch in three places, behavior-preserving for PMI:

### 4.1 `data_loader.py`
```python
# Old
vals = [clean_value(v) for v in col]
vals = [v for v in vals if v != ""]
if len(set(vals)) < 2:
    continue
cleaned.append(vals)

# New — keep "" markers, count unique non-empty
vals = [clean_value(v) for v in col]
if len({v for v in vals if v != ""}) < 2:
    continue
cleaned.append(vals)
```
Apply identically in `_load_jsonl` and `_load_csv_folder`.

### 4.2 `cooccurrence_index.py` and `npmi.py`
```python
# Old
distinct = sorted(set(col))

# New — skip "" sentinel
distinct = sorted(v for v in set(col) if v)
```
Apply in `build_cooccurrence_index` and `compute_coherence`.

### 4.3 Verification
- All 50 existing WP1 tests must still pass after the patch.
- Re-running `main.py` on `data/sample.json` must produce identical totals (column count, threshold sweep, top/bottom scores). The patch is a no-op for PMI; we move the empty-filter from the loader to the index/coherence functions.
- Add **one** new test `test_jsonl_loader_preserves_row_alignment` confirming `[["a","","c"], ["x","y","z"]]` produces two columns of length 3.

## 5. Architecture

```
output/filtered_corpus.jsonl
       │
       ▼
   wp2.py (CLI)
       │
       ▼
   fd_filter.filter_candidates_by_fd
       │  for each filtered table:
       │    for each ordered (i, j), i ≠ j:
       │      X = relation[i], Y = relation[j]
       │      drop rows where X[k]=="" or Y[k]==""
       │      if <3 rows or <2 unique X or <2 unique Y: skip
       │      compute θ via witness-subset construction
       │      if θ ≥ threshold: emit candidate
       ▼
   fd_filter.save_candidates → output/candidates.jsonl
   fd_filter.candidates_summary → stdout
```

WP1 and WP2 are sequential commands:
```bash
python main.py --corpus_path data/sample.json --output_folder output/
python wp2.py  --filtered_corpus output/filtered_corpus.jsonl --output_folder output/
```

WP2 is purely local-per-table — no global index, no caching needed. Reruns at different θ are cheap regardless.

## 6. Module specs

### 6.1 `fd_filter.py`

```python
"""Approximate-FD filtering of column pairs (paper §3.2).

For each table from WP1, enumerate ordered column pairs (C_i, C_j),
i ≠ j. Apply X →_θ Y check per Definition 2; keep pairs with θ ≥ 0.95.
"""

from __future__ import annotations
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
    Returns (theta, surviving_pairs, row_count, covered_rows) where:
      - theta = covered_rows / row_count
      - surviving_pairs is the deduped list of (x, most_common_y),
        one per distinct x
      - row_count = |R| (non-empty rows)
      - covered_rows = |R̄| (rows in the witness subset)
    Returns (0.0, [], 0, 0) if fewer than min_rows non-empty rows or
    either column has fewer than 2 distinct non-empty values.
    """


def filter_candidates_by_fd(
    filtered_records: Iterable[Dict[str, Any]],
    *,
    theta_threshold: float = 0.95,
    min_rows: int = 3,
) -> List[Candidate]:
    """For each filtered table, enumerate ordered column pairs (i, j),
    i ≠ j. Run compute_approx_fd. Keep pairs with theta >= theta_threshold.
    Returns one Candidate dict per surviving pair (schema in §7)."""


def save_candidates(candidates: List[Candidate], output_path: str) -> None:
    """Write JSONL, one candidate per line."""


def candidates_summary(candidates: List[Candidate]) -> None:
    """Print: # candidates, # source tables represented, theta
    distribution (mean / min / max), 5 example top candidates by theta."""
```

### 6.2 `wp2.py`

CLI:
```
--filtered_corpus    default output/filtered_corpus.jsonl
--output_folder      required (or default output/)
--theta              default 0.95
--min_rows           default 3
--output_filename    default candidates.jsonl
```

Behavior:
- Read filtered_corpus.jsonl as a list of dicts.
- Call `filter_candidates_by_fd`.
- Write `<output_folder>/<output_filename>`.
- Print `candidates_summary`.
- Report total elapsed time.

## 7. Output schema

`output/candidates.jsonl` — one JSON object per line:

```jsonc
{
  "pairs": [["united states", "usa"],
            ["canada", "can"],
            ["japan", "jpn"]],          // deduped (l, r) pairs from R̄, one per distinct x
  "theta": 0.971,                       // approximate-FD score, in [theta_threshold, 1.0]
  "row_count": 7,                       // |R| = non-empty rows used
  "covered_rows": 6,                    // |R̄| = witness-subset size
  "source_table_index": 0,              // line number in filtered_corpus.jsonl
  "left_column_index": 0,               // index into the filtered table's `relation`
  "right_column_index": 1,
  "source_metadata": {                  // pass-through from filtered_corpus.jsonl
    "pageTitle": "...",                 // (excluding relation, coherence_scores,
    "url": "...",                       //  rejected_column_indices)
    "tableType": "RELATION",
    "tableNum": 11
  }
}
```

Field rationale and explicit non-fields documented in the brainstorming session; key points:
- `(source_table_index, left_column_index, right_column_index)` is a stable unique key.
- `pairs` is deduplicated (one entry per distinct x). WP3 expects this set form per paper §4.1.
- Both `(C_i, C_j)` and `(C_j, C_i)` may appear if both pass FD (1:1 mappings) or only one (N:1).

## 8. Testing

`tests/test_wp2.py` — pytest. Reuses a new `fd_synthetic_table` fixture in `conftest.py`:

```python
@pytest.fixture
def fd_synthetic_table():
    """Five row-aligned columns, 8 rows. Designed for FD test cases.

    Includes a name-ambiguity case (portland mapping to both oregon and
    maine) and a constant column to exercise edge cases.
    """
    return ({}, [
        ["united", "united", "canada", "japan", "germany", "france", "portland", "portland"],
        ["usa",    "usa",    "can",    "jpn",   "deu",     "fra",    "oregon",   "maine"],
        ["portland","portland","vancouver","tokyo","berlin","paris","",          ""],
        ["yes",    "yes",    "no",     "yes",   "yes",     "no",     "yes",      "no"],
        ["A",      "A",      "A",      "A",     "A",       "A",      "A",        "A"],
    ])
```

Test cases (12 total):

1. `test_perfect_fd_passes` — clean (X, Y) without ambiguity passes at θ=0.95.
2. `test_name_ambiguity_rejected_at_095` — `(LEFT_CC, RIGHT_CC)` whole column has θ ≈ 0.875, rejected at θ=0.95.
3. `test_name_ambiguity_accepted_at_lower_theta` — same pair passes at `--theta 0.85`.
4. `test_constant_column_rejected` — `(LEFT_CC, CONST)` has Y constant — fewer than 2 distinct Y values → rejected.
5. `test_too_few_rows_rejected` — pair where post-empty-filtering leaves fewer than `min_rows`.
6. `test_empty_rows_dropped` — `(LEFT_CC, AMBIG)` with empty AMBIG values; those rows excluded, FD computed on remaining.
7. `test_pairs_are_deduplicated` — identical `(x, y)` rows appear once in `pairs`.
8. `test_surviving_pairs_match_witness_subset` — for each x in `pairs`, the y is the most common y in the original column.
9. `test_ordered_pairs_are_distinct_candidates` — running on a 2-column table yields up to 2 candidates: (0→1) and (1→0).
10. `test_save_candidates_jsonl_schema` — every output line has the 8 required fields with correct types.
11. `test_filter_candidates_passes_through_metadata` — `source_metadata` carries `pageTitle`/`url`/etc., excludes WP1-added fields.
12. `test_wp2_end_to_end_smoke` — subprocess invocation of `wp2.py` produces `candidates.jsonl` with expected structure.

Plus **one** new test in `tests/test_wp1.py`:
- `test_jsonl_loader_preserves_row_alignment` — confirms `[["a","","c"], ["x","y","z"]]` produces two columns of length 3.

**Total expected:** 51 (WP1) + 12 (WP2) = 63 tests.

No mocks. Real math on real synthetic data.

## 9. Doc updates

### `claude/deviations.md`
Append a new top-level section "WP2 — Section 3.2 (Column-Pair Filtering by FD)" documenting design choices that aren't obvious from code:
- WP2 was designed against the paper directly (no WP2 prompt existed).
- JSONL output with deduped `(l, r)` pairs.
- WP1 row-alignment patch.
- `min_rows=3` and `<2 distinct values` rejection.
- Greedy witness-subset construction (provably optimal).
- No reverse-direction inference; ordered pairs evaluated independently.

### `USAGE.md`
- Add a section "Running WP2" with the `wp2.py` command and flags.
- Add a section "Interpreting `candidates.jsonl`" describing each field, how to read θ, what `pairs` represents, and a few Python snippets for inspecting the output (top candidates by θ, candidates per source table, browsing one candidate).
- Add guidance on choosing θ.
- Update "How it works" to mention Section 3 is now fully implemented.

### `WALKTHROUGH.md`
- New top-level section "WP2: Column-pair filtering by FD".
- Cover the problem (most ordered column pairs aren't real mappings), the math (Definition 2, witness-subset construction, worked Portland example), the data flow extension, why ordered pairs matter (1:1 vs N:1).
- Extend "If someone asks you" Q&A with FD-specific questions.
- Update "What's NOT in WP1" → "What's NOT in WP1+WP2"; remove FD filtering, leave WP3/WP4.

## 10. Constraints

- Python 3.8+
- Allowed libs: stdlib only (`collections`, `itertools`, `json`, `os`, `sys`, `argparse`, `time`, `typing`). No new package dependencies.
- Every public function: type hints + docstring.
- Every module: module-level docstring.
- Print progress and timings.

## 11. Out of scope

- WP3 (paper §4): table synthesis via positive-compatibility + FD-conflict graph optimization.
- WP4 (paper §5): conflict resolution.
- Streaming / out-of-core processing for >sample-scale corpora.
- Distributed execution.
- Bidirectional FD inference (treating `X →_θ Y` and `Y →_θ X` as a single 1:1 mapping). Each direction stays independent in the output; WP3 can later detect 1:1 by finding both directions present.

## 12. Deliverables

```
fd_filter.py                                   (new)
wp2.py                                         (new)
tests/test_wp2.py                              (new)
data_loader.py                                 (patch — preserve row alignment)
cooccurrence_index.py                          (patch — skip "" in distinct)
npmi.py                                        (patch — skip "" in distinct)
tests/test_wp1.py                              (1 new test for row alignment)
conftest.py                                    (1 new fixture: fd_synthetic_table)
claude/deviations.md                           (new WP2 section)
USAGE.md                                       (new WP2 section)
WALKTHROUGH.md                                 (new WP2 section + updated framing)
```

Run command:
```bash
python wp2.py --filtered_corpus output/filtered_corpus.jsonl \
              --output_folder output/ \
              --theta 0.95
```
