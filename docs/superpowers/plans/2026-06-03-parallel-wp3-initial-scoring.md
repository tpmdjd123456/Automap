# Parallel WP3 Initial Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize Step 2 of `greedy_partition` (initial compatibility-graph scoring) — the ~90% of WP3 wall time — using `multiprocessing.Pool` with a per-worker initializer, while keeping bit-identical output to the existing sequential path.

**Architecture:** Refactor `synthesis.py` to extract three private helpers (`_build_overlap_set`, `_compute_initial_scores`, `_run_merge_loop`) so `greedy_partition` becomes pure orchestration. Add sibling functions `parallel_compute_initial_scores` and `parallel_greedy_partition` to `parallel_pipeline.py`, following the existing `parallel_score_corpus` / `parallel_fd_filter` pattern. Workers receive `candidates` once via a top-level `initializer`; each task sends only `(ci, cj)`. Determinism is preserved by iterating `sorted(overlapping_pairs)` in both paths. Wire in via `--parallel_workers N` in `main.py`.

**Tech Stack:** Python 3.10+, `multiprocessing.Pool`, `tqdm` (already in synthesis.py), pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-parallel-wp3-initial-scoring-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `conftest.py` | Modify | Add `synthesis_candidates` fixture — hand-crafted 8 candidates exercising identity, overlap, conflict, blocked-by-tau, score-tie, isolation. |
| `synthesis.py` | Modify | Extract `_build_overlap_set`, `_compute_initial_scores`, `_run_merge_loop`. `greedy_partition` becomes orchestration. No behavior change. |
| `parallel_pipeline.py` | Modify | Add `_CANDIDATES`/`_USE_APPROX` module globals, `_init_scoring_worker`, `_score_edge_worker`, `parallel_compute_initial_scores`, `parallel_greedy_partition`. |
| `tests/test_parallel_synthesis.py` | Create | All synthesis-parallel tests (equivalence, n_workers=1, edge cases, determinism, refactor regression). |
| `main.py` | Modify | Add `--parallel_workers` CLI flag; route Stage 6 through the parallel path when `>1`. |

---

## Task 1: Add `synthesis_candidates` fixture

**Files:**
- Modify: `conftest.py`

The fixture exercises every interesting interaction the tests rely on: a perfect-match (will merge), a partial overlap, an isolated singleton, and a conflict (high overlap on left value but different rights — blocked by negative score).

- [ ] **Step 1: Read existing conftest.py to find an insertion point**

Run: `cat conftest.py | head -40`
Expected: see existing fixtures (`synthetic_corpus` etc.). Add new fixture below them.

- [ ] **Step 2: Append the fixture**

Add to the end of `conftest.py`:

```python
@pytest.fixture
def synthesis_candidates():
    """Hand-crafted 8 candidates for synthesis tests.

    Designed to exercise:
      - perfect merge (0 ↔ 3, identical pairs)
      - partial overlap (0 ↔ 1, two shared pairs)
      - conflict / negative score (0 ↔ 2, same lefts, different rights)
      - isolated singletons (4, 7 — share nothing)
      - identity / degenerate (5 — left == right per pair)
      - rank-numeric (6 — numeric left, won't merge)
      - single-edge overlap (0 ↔ 6 share zero, 0 ↔ 1 share two)

    Each candidate is a dict matching what `load_candidates` returns:
    `{"pairs": [(l, r), ...], "theta": float, "row_count": int,
      "covered_rows": int, "source_table_index": int,
      "left_column_index": int, "right_column_index": int,
      "source_metadata": {}}`.
    """
    def mk(idx, pairs):
        return {
            "pairs": [tuple(p) for p in pairs],
            "theta": 1.0,
            "row_count": len(pairs),
            "covered_rows": len(pairs),
            "source_table_index": idx,
            "left_column_index": 0,
            "right_column_index": 1,
            "source_metadata": {},
        }
    return [
        mk(0, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell")]),
        mk(1, [("dune", "herbert"), ("ender", "card"), ("foundation", "asimov")]),
        mk(2, [("dune", "1965"), ("foundation", "1951"), ("1984", "1949")]),
        mk(3, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell")]),
        mk(4, [("france", "paris"), ("japan", "tokyo"), ("egypt", "cairo")]),
        mk(5, [("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma")]),
        mk(6, [("1", "100"), ("2", "200"), ("3", "300")]),
        mk(7, [("red", "warm"), ("blue", "cool"), ("green", "neutral")]),
    ]
```

- [ ] **Step 3: Verify it imports**

Run: `pytest --collect-only conftest.py 2>&1 | tail -3`
Expected: no collection error.

- [ ] **Step 4: Commit**

```bash
git add conftest.py
git commit -m "Add synthesis_candidates fixture for WP3 parallel tests"
```

---

## Task 2: Pin current `greedy_partition` behavior with a regression test

This test captures the *current* sequential output so the refactor in Tasks 3-5 can't silently change it. Expected output is computed once by running `greedy_partition` on the fixture and pasted into the test as a constant.

**Files:**
- Create: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Capture expected output**

Run interactively in the repo root with the venv active:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "tests")
from conftest import synthesis_candidates
from synthesis import greedy_partition
# Re-build the fixture (pytest fixtures aren't directly callable)
def mk(idx, pairs):
    return {"pairs":[tuple(p) for p in pairs],"theta":1.0,"row_count":len(pairs),
            "covered_rows":len(pairs),"source_table_index":idx,
            "left_column_index":0,"right_column_index":1,"source_metadata":{}}
cands = [
    mk(0,[("dune","herbert"),("foundation","asimov"),("1984","orwell")]),
    mk(1,[("dune","herbert"),("ender","card"),("foundation","asimov")]),
    mk(2,[("dune","1965"),("foundation","1951"),("1984","1949")]),
    mk(3,[("dune","herbert"),("foundation","asimov"),("1984","orwell")]),
    mk(4,[("france","paris"),("japan","tokyo"),("egypt","cairo")]),
    mk(5,[("alpha","alpha"),("beta","beta"),("gamma","gamma")]),
    mk(6,[("1","100"),("2","200"),("3","300")]),
    mk(7,[("red","warm"),("blue","cool"),("green","neutral")]),
]
parts = greedy_partition(cands, tau=-0.2, theta_overlap=1, use_approx=True, output_folder="/tmp")
print("EXPECTED =", repr(sorted([sorted(p) for p in parts])))
PY
```

Capture the printed `EXPECTED = [...]` line — paste it into the test in Step 2.

- [ ] **Step 2: Write the regression test**

Create `tests/test_parallel_synthesis.py`:

```python
"""Tests for parallel WP3 (initial-scoring) — see
docs/superpowers/specs/2026-06-03-parallel-wp3-initial-scoring-design.md."""

import pytest
from synthesis import greedy_partition

# Captured from a sequential run of greedy_partition on the
# synthesis_candidates fixture BEFORE the refactor. This pins the current
# behavior so the refactor in synthesis.py cannot silently change output.
# Generated by the one-off script in Task 2 Step 1.
EXPECTED_PARTITIONS = ...  # paste captured value here, e.g. [[0,1,3],[2],[4],[5],[6],[7]]


def _canonical(partitions):
    """Return partitions in a stable comparison form."""
    return sorted(sorted(p) for p in partitions)


def test_refactor_preserves_behavior(synthesis_candidates):
    """greedy_partition on the fixture must keep producing EXPECTED_PARTITIONS."""
    actual = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        output_folder="/tmp",
    )
    assert _canonical(actual) == EXPECTED_PARTITIONS
```

- [ ] **Step 3: Run the test, verify it passes today**

Run: `pytest tests/test_parallel_synthesis.py::test_refactor_preserves_behavior -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Pin greedy_partition behavior with regression test on synthesis fixture"
```

---

## Task 3: Refactor — extract `_build_overlap_set` from `greedy_partition`

**Files:**
- Modify: `synthesis.py`

Today (in `greedy_partition`) lines roughly 318-336 build `overlapping_pairs` via two nested loops over `pair_index` and `left_index`. Extract that into a helper.

- [ ] **Step 1: Read the current overlap-building block**

Run: `sed -n '315,340p' synthesis.py`
Expected: see the two `for indices in pair_index.values(): ... for indices in left_index.values(): ...` blocks.

- [ ] **Step 2: Add the helper above `greedy_partition`**

In `synthesis.py`, immediately above `def greedy_partition(`, add:

```python
def _build_overlap_set(
    pair_index: Dict[Tuple[str, str], List[int]],
    left_index: Dict[str, List[int]],
) -> Set[Tuple[int, int]]:
    """Union of all candidate-index pairs that share either an exact (l,r)
    pair (via pair_index) or an exact left value (via left_index).
    Returned as ordered tuples (a, b) with a < b.
    """
    overlapping: Set[Tuple[int, int]] = set()
    for indices in pair_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = indices[x], indices[y]
                if a > b:
                    a, b = b, a
                overlapping.add((a, b))
    for indices in left_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = indices[x], indices[y]
                if a > b:
                    a, b = b, a
                overlapping.add((a, b))
    return overlapping
```

- [ ] **Step 3: Replace the inline block in `greedy_partition`**

In `greedy_partition`, replace the two `for indices in pair_index.values(): ...` and `for indices in left_index.values(): ...` blocks (and the `overlapping_pairs: Set[...] = set()` initializer above them) with:

```python
    overlapping_pairs = _build_overlap_set(pair_index, left_index)
```

- [ ] **Step 4: Run the regression test**

Run: `pytest tests/test_parallel_synthesis.py::test_refactor_preserves_behavior -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add synthesis.py
git commit -m "Extract _build_overlap_set helper from greedy_partition"
```

---

## Task 4: Refactor — extract `_compute_initial_scores` (with sorted iteration for determinism)

**Files:**
- Modify: `synthesis.py`

Today (lines roughly 345-361) the initial pos/neg scores are computed by iterating `overlapping_pairs` (set order — nondeterministic). Extract into a helper and **iterate `sorted(overlapping_pairs)`** so sequential and parallel paths share the same insertion order in the resulting dicts (Determinism Contract §1 in the spec).

- [ ] **Step 1: Read the current scoring block**

Run: `sed -n '343,365p' synthesis.py`
Expected: see the loop computing `ps = positive_score(...)`, `ns = negative_score(...)`, etc. Note: this block also increments `positive_edges` and `blocking_edges` for the print summary — those counts move with the helper.

- [ ] **Step 2: Add the helper above `greedy_partition`**

In `synthesis.py`, below `_build_overlap_set`, add:

```python
def _compute_initial_scores(
    overlapping_pairs: Set[Tuple[int, int]],
    candidates: List[Candidate],
    use_approx: bool,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], int, int]:
    """Compute positive and negative scores for every overlap edge.

    Iterates `sorted(overlapping_pairs)` so the resulting dicts have
    deterministic insertion order — important because the greedy merge
    loop scans `pos_scores.items()` to find the maximum, and tie-breaks
    depend on iteration order.

    Returns: (pos_scores, neg_scores, positive_edges, blocking_edges).
    `blocking_edges` here is the count where `ns < -0.2` — kept for
    parity with the existing summary print. The merge loop applies the
    real `tau` threshold itself.
    """
    pos_scores: Dict[Tuple[int, int], float] = {}
    neg_scores: Dict[Tuple[int, int], float] = {}
    positive_edges = 0
    blocking_edges = 0
    for ci, cj in sorted(overlapping_pairs):
        key = (ci, cj)
        bp = list(candidates[ci]["pairs"])
        bq = list(candidates[cj]["pairs"])
        ps = positive_score(bp, bq, use_approx=use_approx)
        ns = negative_score(bp, bq, use_approx=use_approx)
        if ps > 0:
            pos_scores[key] = ps
            positive_edges += 1
        if ns < 0:
            neg_scores[key] = ns
            if ns < -0.2:
                blocking_edges += 1
    return pos_scores, neg_scores, positive_edges, blocking_edges
```

- [ ] **Step 3: Replace the inline block in `greedy_partition`**

Replace the `pos_scores: Dict... = {}` initialization through the `print(f"    Blocking negative edges...")` line (the entire scoring loop plus the two summary prints' inputs) with:

```python
    pos_scores, neg_scores, positive_edges, blocking_edges = _compute_initial_scores(
        overlapping_pairs, candidates, use_approx
    )
    print(f"    Non-zero positive edges: {positive_edges}")
    print(f"    Blocking negative edges (w- < tau): {blocking_edges}")
    print(f"  Running greedy partitioning...")
```

(The two print lines and the "Running greedy partitioning..." print already exist after the scoring loop — keep them. Just remove the loop itself.)

- [ ] **Step 4: Run the regression test**

Run: `pytest tests/test_parallel_synthesis.py::test_refactor_preserves_behavior -v`
Expected: PASS. *Note:* output may technically differ from the previous behavior on score ties because sequential now iterates `sorted`. If the test fails, this means a tie-break diverged — re-run Task 2 Step 1 to recapture `EXPECTED_PARTITIONS` from the refactored (sorted) sequential, then re-run this test.

- [ ] **Step 5: Commit**

```bash
git add synthesis.py tests/test_parallel_synthesis.py
git commit -m "Extract _compute_initial_scores with deterministic sorted iteration"
```

---

## Task 5: Refactor — extract `_run_merge_loop` from `greedy_partition`

**Files:**
- Modify: `synthesis.py`

Today (lines roughly 363-end of function) the merge loop is the bulk of `greedy_partition`. Extract it. After this task, `greedy_partition` is pure orchestration: build index → build overlap set → score → run merge loop.

- [ ] **Step 1: Read the current merge loop**

Run: `sed -n '362,500p' synthesis.py`
Expected: see the `merge_count = 0`, `while True:` loop and everything after through the `return [sorted(members) ...]`.

- [ ] **Step 2: Add the helper above `greedy_partition`**

In `synthesis.py`, below `_compute_initial_scores`, add:

```python
def _run_merge_loop(
    candidates: List[Candidate],
    pos_scores: Dict[Tuple[int, int], float],
    neg_scores: Dict[Tuple[int, int], float],
    tau: float,
    theta_overlap: int,
    output_folder: str,
) -> List[Partition]:
    """The greedy merge loop. Mutates `pos_scores`/`neg_scores` as
    partitions merge. Writes `computed_edge_scores.jsonl` to
    `output_folder` (as in the existing code).
    """
    n = len(candidates)
    part_members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    part_pairs: Dict[int, List[Tuple[str, str]]] = {
        i: list(candidates[i]["pairs"]) for i in range(n)
    }
    next_pid = n
    # Move the full existing merge-loop body here (everything from
    # `merge_count = 0` through `return [sorted(members) for members in
    # part_members.values()]`). Adjust by removing the duplicate
    # `part_members`/`part_pairs`/`next_pid` initialization that already
    # appears at the top of `greedy_partition` today — we kept those
    # there but the helper now owns them.
    ...
```

**Note:** the `...` above stands for the existing merge-loop body — preserve it byte-for-byte. The only edit is moving it into this function and removing the now-duplicate setup lines from `greedy_partition`.

- [ ] **Step 3: Simplify `greedy_partition` to pure orchestration**

Replace `greedy_partition`'s body (everything after the docstring) with:

```python
    n = len(candidates)
    if n == 0:
        return []

    print(f"  Building inverted index...")
    pair_index, left_index = build_inverted_index(candidates)
    print(f"    pair_index: {len(pair_index)} unique pairs")
    print(f"    left_index: {len(left_index)} unique left values")

    print(f"  Computing initial compatibility graph...")
    overlapping_pairs = _build_overlap_set(pair_index, left_index)
    pos_scores, neg_scores, positive_edges, blocking_edges = _compute_initial_scores(
        overlapping_pairs, candidates, use_approx
    )
    print(f"    Non-zero positive edges: {positive_edges}")
    print(f"    Blocking negative edges (w- < tau): {blocking_edges}")
    print(f"  Running greedy partitioning...")

    return _run_merge_loop(
        candidates, pos_scores, neg_scores, tau, theta_overlap, output_folder
    )
```

- [ ] **Step 4: Run the regression test**

Run: `pytest tests/test_parallel_synthesis.py::test_refactor_preserves_behavior -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add synthesis.py
git commit -m "Extract _run_merge_loop; greedy_partition now pure orchestration"
```

---

## Task 6: Add parallel worker infrastructure to `parallel_pipeline.py`

**Files:**
- Modify: `parallel_pipeline.py`

Add the module globals, initializer, and worker function. No public API yet — that lands in Task 7.

- [ ] **Step 1: Read the top of `parallel_pipeline.py`**

Run: `sed -n '1,35p' parallel_pipeline.py`
Expected: see existing imports and worker-function comment block.

- [ ] **Step 2: Add imports for the synthesis worker**

In `parallel_pipeline.py`, near the existing imports (around line 28), add:

```python
from synthesis import positive_score, negative_score
```

- [ ] **Step 3: Add module globals and worker functions**

Below the existing `_fd_filter_worker` function, before the next section comment, add:

```python
# ---------------------------------------------------------------------------
# Parallel WP3 initial scoring (see specs/2026-06-03-parallel-wp3-...)
# ---------------------------------------------------------------------------

# Module-level globals populated in each worker process by
# `_init_scoring_worker`. They stay None in the parent.
_CANDIDATES = None
_USE_APPROX = None


def _init_scoring_worker(candidates, use_approx):
    """Pool initializer: stash candidates and use_approx in worker globals
    so per-task args can be just `(ci, cj)`."""
    global _CANDIDATES, _USE_APPROX
    _CANDIDATES = candidates
    _USE_APPROX = use_approx


def _score_edge_worker(edge):
    """Score one overlap edge. Reads from module globals set by
    `_init_scoring_worker`. Returns `(ci, cj, pos, neg)`."""
    ci, cj = edge
    bp = list(_CANDIDATES[ci]["pairs"])
    bq = list(_CANDIDATES[cj]["pairs"])
    ps = positive_score(bp, bq, use_approx=_USE_APPROX)
    ns = negative_score(bp, bq, use_approx=_USE_APPROX)
    return ci, cj, ps, ns
```

- [ ] **Step 4: Verify imports resolve**

Run: `python -c "import parallel_pipeline; print('ok')"`
Expected: `ok` printed, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add parallel_pipeline.py
git commit -m "Add parallel scoring worker infrastructure to parallel_pipeline"
```

---

## Task 7: Add `parallel_compute_initial_scores` and `parallel_greedy_partition`

**Files:**
- Modify: `parallel_pipeline.py`

- [ ] **Step 1: Append the public functions**

At the bottom of `parallel_pipeline.py` (above the existing `benchmark()` section, or after it — order is fine), add:

```python
def parallel_compute_initial_scores(
    overlapping_pairs,
    candidates,
    use_approx,
    n_workers=None,
    chunk_size=1000,
):
    """Parallel drop-in for synthesis._compute_initial_scores.

    Returns (pos_scores, neg_scores, positive_edges, blocking_edges)
    — same shape as the sequential helper, bit-identical output.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    edges = sorted(overlapping_pairs)  # deterministic order
    pos_scores = {}
    neg_scores = {}
    positive_edges = 0
    blocking_edges = 0

    with mp.Pool(
        processes=n_workers,
        initializer=_init_scoring_worker,
        initargs=(candidates, use_approx),
    ) as pool:
        for ci, cj, ps, ns in pool.imap(
            _score_edge_worker, edges, chunksize=chunk_size
        ):
            key = (ci, cj)
            if ps > 0:
                pos_scores[key] = ps
                positive_edges += 1
            if ns < 0:
                neg_scores[key] = ns
                if ns < -0.2:
                    blocking_edges += 1

    return pos_scores, neg_scores, positive_edges, blocking_edges


def parallel_greedy_partition(
    candidates,
    tau=-0.2,
    theta_overlap=1,
    use_approx=True,
    n_workers=None,
    chunk_size=1000,
    output_folder="output",
):
    """Parallel sibling of synthesis.greedy_partition.

    Identical orchestration; the initial-score computation is parallelized.
    Output is bit-identical to greedy_partition for any input.
    """
    from synthesis import build_inverted_index, _build_overlap_set, _run_merge_loop

    n = len(candidates)
    if n == 0:
        return []

    print(f"  Building inverted index...")
    pair_index, left_index = build_inverted_index(candidates)
    print(f"    pair_index: {len(pair_index)} unique pairs")
    print(f"    left_index: {len(left_index)} unique left values")

    print(f"  Computing initial compatibility graph (parallel, {n_workers or mp.cpu_count()} workers)...")
    overlapping_pairs = _build_overlap_set(pair_index, left_index)
    pos_scores, neg_scores, positive_edges, blocking_edges = parallel_compute_initial_scores(
        overlapping_pairs, candidates, use_approx,
        n_workers=n_workers, chunk_size=chunk_size,
    )
    print(f"    Non-zero positive edges: {positive_edges}")
    print(f"    Blocking negative edges (w- < tau): {blocking_edges}")
    print(f"  Running greedy partitioning...")

    return _run_merge_loop(
        candidates, pos_scores, neg_scores, tau, theta_overlap, output_folder
    )
```

- [ ] **Step 2: Verify imports resolve**

Run: `python -c "from parallel_pipeline import parallel_greedy_partition, parallel_compute_initial_scores; print('ok')"`
Expected: `ok` printed.

- [ ] **Step 3: Commit**

```bash
git add parallel_pipeline.py
git commit -m "Add parallel_compute_initial_scores and parallel_greedy_partition"
```

---

## Task 8: Write the critical equivalence test

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Append the test**

Add to `tests/test_parallel_synthesis.py`:

```python
from parallel_pipeline import parallel_greedy_partition


@pytest.mark.parametrize("n_workers", [2, 4])
def test_parallel_equals_sequential(synthesis_candidates, n_workers):
    """parallel_greedy_partition must produce the same partitions as
    greedy_partition for any input — bit-identical, no tie-break drift."""
    seq = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        output_folder="/tmp",
    )
    par = parallel_greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        n_workers=n_workers, chunk_size=2,
        output_folder="/tmp",
    )
    assert _canonical(par) == _canonical(seq)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_parallel_synthesis.py::test_parallel_equals_sequential -v`
Expected: PASS for both `n_workers=2` and `n_workers=4`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Test parallel_greedy_partition matches sequential output"
```

---

## Task 9: Test `n_workers=1` path

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

Catches degenerate-case bugs (e.g., Pool with a single worker doesn't deadlock or short-circuit incorrectly).

- [ ] **Step 1: Append the test**

```python
def test_parallel_n_workers_one(synthesis_candidates):
    """n_workers=1 routes through the parallel code path (still uses a
    Pool) and produces the same output."""
    seq = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        output_folder="/tmp",
    )
    par = parallel_greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        n_workers=1, chunk_size=2,
        output_folder="/tmp",
    )
    assert _canonical(par) == _canonical(seq)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_parallel_synthesis.py::test_parallel_n_workers_one -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Test parallel_greedy_partition with n_workers=1"
```

---

## Task 10: Edge-case tests

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Append the tests**

```python
def _trivial(idx, pairs):
    return {"pairs":[tuple(p) for p in pairs],"theta":1.0,
            "row_count":len(pairs),"covered_rows":len(pairs),
            "source_table_index":idx,"left_column_index":0,
            "right_column_index":1,"source_metadata":{}}


def test_parallel_empty():
    """Zero candidates → empty partition list, no Pool created."""
    assert parallel_greedy_partition([], n_workers=2, output_folder="/tmp") == []


def test_parallel_single_candidate():
    """One candidate → one singleton partition."""
    cands = [_trivial(0, [("a", "x"), ("b", "y"), ("c", "z")])]
    result = parallel_greedy_partition(cands, n_workers=2, output_folder="/tmp")
    assert _canonical(result) == [[0]]


def test_parallel_no_overlap():
    """Candidates with no shared pairs or left values → all singletons."""
    cands = [
        _trivial(0, [("a", "x"), ("b", "y"), ("c", "z")]),
        _trivial(1, [("d", "p"), ("e", "q"), ("f", "r")]),
        _trivial(2, [("g", "m"), ("h", "n"), ("i", "o")]),
    ]
    seq = greedy_partition(cands, output_folder="/tmp")
    par = parallel_greedy_partition(cands, n_workers=2, output_folder="/tmp")
    assert _canonical(par) == _canonical(seq) == [[0], [1], [2]]
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_parallel_synthesis.py -k "empty or single or no_overlap" -v`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Test parallel_greedy_partition edge cases: empty, single, no-overlap"
```

---

## Task 11: Determinism test (same input twice → same output)

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Append the test**

```python
def test_parallel_determinism(synthesis_candidates):
    """Running the parallel path twice on identical input produces
    identical output (independent from parallel-vs-sequential equality)."""
    run1 = parallel_greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        n_workers=4, chunk_size=2, output_folder="/tmp",
    )
    run2 = parallel_greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        n_workers=4, chunk_size=2, output_folder="/tmp",
    )
    assert _canonical(run1) == _canonical(run2)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_parallel_synthesis.py::test_parallel_determinism -v`
Expected: PASS.

- [ ] **Step 3: Run the full test file as a final check**

Run: `pytest tests/test_parallel_synthesis.py -v`
Expected: all PASS (8 tests: refactor regression, equivalence × 2, n=1, empty, single, no_overlap, determinism).

- [ ] **Step 4: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Test parallel_greedy_partition re-run determinism"
```

---

## Task 12: Wire `--parallel_workers` into `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the CLI flag**

In `main.py`'s `parse_args()`, after the existing `--theta_overlap` argument, add:

```python
    p.add_argument("--parallel_workers", type=int, default=1,
                   help="Run WP3 initial scoring in parallel with N workers "
                        "(default 1 = sequential). Recommended on dama: 14 "
                        "(one per physical core); on laptop: 6.")
```

- [ ] **Step 2: Replace the Stage 6 `greedy_partition` call**

Find the call in `main()` that looks like:

```python
    partitions = greedy_partition(
        wp3_candidates,
        tau=args.tau,
        theta_overlap=args.theta_overlap,
        use_approx=not args.no_approx,
        output_folder=args.output_folder
    )
```

Replace with:

```python
    if args.parallel_workers > 1:
        from parallel_pipeline import parallel_greedy_partition
        partitions = parallel_greedy_partition(
            wp3_candidates,
            tau=args.tau,
            theta_overlap=args.theta_overlap,
            use_approx=not args.no_approx,
            n_workers=args.parallel_workers,
            output_folder=args.output_folder,
        )
    else:
        partitions = greedy_partition(
            wp3_candidates,
            tau=args.tau,
            theta_overlap=args.theta_overlap,
            use_approx=not args.no_approx,
            output_folder=args.output_folder,
        )
```

- [ ] **Step 3: Smoke-test from CLI help**

Run: `python main.py --help 2>&1 | grep -A1 parallel_workers`
Expected: the new flag and its help text appear.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "Add --parallel_workers flag wiring WP3 to parallel_greedy_partition"
```

---

## Task 13: Manual verification on dama (post-merge)

This is a **manual verification step**, not a unit test. Run it after Task 12 is merged. It confirms the real win on production-scale data.

- [ ] **Step 1: Sync code to dama**

Run from repo root locally:

```bash
rsync -az \
  --exclude '.git' --exclude '.venv' --exclude 'output' --exclude 'data' \
  --exclude '__pycache__' --exclude '.idea' --exclude '*.ipynb' --exclude 'papers' \
  --exclude 'claude' --exclude '.DS_Store' \
  ./ dama:Automap/
```

- [ ] **Step 2: Launch parallel run reusing the existing WDC candidates**

The existing `output/candidates.jsonl` on dama has the 33,283 WDC candidates. To avoid re-running WP1+WP2, run a small one-off Python that loads them and calls `parallel_greedy_partition` directly. From dama:

```bash
ssh dama 'cd ~/Automap && nohup .venv/bin/python -u -c "
from synthesis import load_candidates, save_synthesized_mappings, synthesis_report
from parallel_pipeline import parallel_greedy_partition
import time
cands = load_candidates(\"output/candidates.jsonl\")
print(f\"Loaded {len(cands)} candidates\", flush=True)
t0 = time.time()
parts = parallel_greedy_partition(cands, n_workers=14, output_folder=\"output/parallel14\")
print(f\"DONE in {time.time()-t0:.1f}s; {len(parts)} partitions\", flush=True)
import os; os.makedirs(\"output/parallel14\", exist_ok=True)
save_synthesized_mappings(parts, cands, \"output/parallel14/synthesized_mappings.jsonl\")
synthesis_report(parts, cands)
" > run_parallel14.log 2>&1 < /dev/null & echo "PID $!"'
```

- [ ] **Step 3: Monitor and compare**

After completion (expected ~30-40 min wall-clock):

```bash
ssh dama 'cd ~/Automap && diff <(sort output/synthesized_mappings.jsonl) <(sort output/parallel14/synthesized_mappings.jsonl) | head -20 && echo "diff exit $?"'
```

Expected: no diff (exit 0) — proves real-world bit-identical output.

- [ ] **Step 4: Record the wall-clock number**

Capture from `run_parallel14.log` the `DONE in <X>s` line and note it in the spec's "Expected speedup" section or a follow-up commit message. This validates the design's ~30 min estimate.

---

## Self-Review Notes

**Spec coverage check:**
- Architecture / Refactor in synthesis.py — Tasks 3, 4, 5 ✓
- Architecture / Additions in parallel_pipeline.py — Tasks 6, 7 ✓
- Worker model & data flow — implemented in Tasks 6, 7 ✓
- Determinism contract (sorted iteration) — Task 4 Step 2 ✓
- main.py integration (`--parallel_workers`) — Task 12 ✓
- Testing — Tasks 2 (regression), 8 (equivalence), 9 (n=1), 10 (edge cases), 11 (determinism) ✓
- Recommended worker counts — documented in main.py flag help (Task 12) ✓
- Manual verification — Task 13 ✓
- Out of scope (heap, lazy negative, within-iteration parallelism) — not touched ✓

**Placeholder scan:** the `...` in Task 5 Step 2 is an intentional stand-in for "preserve existing merge-loop body byte-for-byte." No vague "TODO" / "handle edge cases" anywhere. The `EXPECTED = ...` in Task 2 Step 2 is explicitly captured in Step 1 first.

**Type/name consistency:** `_build_overlap_set`, `_compute_initial_scores`, `_run_merge_loop`, `_init_scoring_worker`, `_score_edge_worker`, `parallel_compute_initial_scores`, `parallel_greedy_partition`, `--parallel_workers` — same identifiers used throughout.
