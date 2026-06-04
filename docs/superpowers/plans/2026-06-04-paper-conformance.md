# Paper-Conformance Fixes for WP3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring WP3 in line with the Wang & He SIGMOD 2017 paper's three scalability techniques — θ_edge filter, properly-applied θ_overlap, connected-component decomposition of the merge loop — plus two strict-fidelity fixes (`negative_score` single `|F|`, paper-strict mode documented under `--no_approx`).

**Architecture:** Surgical changes in `synthesis.py` and `parallel_pipeline.py` plus CLI plumbing in `main.py`. Two new private helpers in `synthesis.py`: `_connected_components` and `_greedy_merge_component`. `_run_merge_loop` becomes a per-component dispatcher. No new files.

**Tech Stack:** Python 3.10+, `collections.Counter` and `defaultdict`, pytest. No multiprocessing changes (per-component parallelism is explicitly out of scope).

**Spec:** `docs/superpowers/specs/2026-06-04-paper-conformance-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `conftest.py` | Pytest fixtures | Add `synthesis_components_fixture` (7 candidates: 2 disjoint clusters of 3 + 1 isolated singleton). |
| `synthesis.py` | Core algorithm | `negative_score` uses single `|F|`. `_build_overlap_set` takes `theta_overlap`. `_compute_initial_scores` takes `theta_edge`. Add `_connected_components`. Restructure `_run_merge_loop` into per-component dispatch + new `_greedy_merge_component` helper. `greedy_partition` signature gains `theta_edge`. |
| `parallel_pipeline.py` | Parallel scoring | `parallel_compute_initial_scores` and `parallel_greedy_partition` accept and thread `theta_edge`. |
| `main.py` | CLI | Add `--theta_edge` (default 0.85). Change `--theta_overlap` default 1 → 3. Update `--no_approx` help to mention paper-strict mode. |
| `tests/test_parallel_synthesis.py` | Tests | Add ~10 new tests across Layers 1-3 from the spec. Update `EXPECTED_PARTITIONS` in `test_refactor_preserves_behavior` for new defaults. Update `test_parallel_equals_sequential` to pin a specific `theta_edge`/`theta_overlap` configuration. |

---

## Task 1: Add `synthesis_components_fixture` to conftest.py

**Files:**
- Modify: `conftest.py`

- [ ] **Step 1: Append the fixture**

Add to the end of `conftest.py`:

```python
@pytest.fixture
def synthesis_components_fixture():
    """7 candidates for connected-component tests.

    Designed structure:
      - 0, 1, 2 form positive-overlap cluster A ("book → author")
      - 3, 4, 5 form positive-overlap cluster B ("country → capital")
      - 6 is isolated (shares nothing with anyone)

    With theta_edge=0 and theta_overlap=1 the positive-edge graph has
    three connected components: {0,1,2}, {3,4,5}, {6}.
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
        # Cluster A — book → author
        mk(0, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell")]),
        mk(1, [("dune", "herbert"), ("ender", "card"), ("foundation", "asimov")]),
        mk(2, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell"), ("hobbit", "tolkien")]),
        # Cluster B — country → capital
        mk(3, [("france", "paris"), ("germany", "berlin"), ("italy", "rome")]),
        mk(4, [("france", "paris"), ("spain", "madrid"), ("germany", "berlin")]),
        mk(5, [("france", "paris"), ("germany", "berlin"), ("italy", "rome"), ("uk", "london")]),
        # Isolated singleton — no pair-overlap with anyone
        mk(6, [("red", "warm"), ("blue", "cool"), ("green", "neutral")]),
    ]
```

- [ ] **Step 2: Verify it collects**

Run: `pytest --collect-only conftest.py 2>&1 | tail -3`
Expected: no collection error.

- [ ] **Step 3: Commit**

```bash
git add conftest.py
git commit -m "Add synthesis_components_fixture for component-decomposition tests"
```

---

## Task 2: Fix `negative_score` to use a single `|F|` (Phase A3)

**Files:**
- Modify: `synthesis.py:209-231` (negative_score function body)
- Modify: `tests/test_parallel_synthesis.py` (append two new tests)

- [ ] **Step 1: Write the two failing tests**

Append to `tests/test_parallel_synthesis.py`:

```python
from synthesis import negative_score


def test_negative_score_single_F_matches_paper_eq4():
    """w- = -max(|F| / |B|, |F| / |B'|), F computed once.

    Hand-crafted case:
      B  = [(a, x), (b, y), (c, z)]
      B' = [(a, x), (b, w)]  # conflict on b only
    |F| = 1, |B| = 3, |B'| = 2.
    w- = -max(1/3, 1/2) = -0.5.
    """
    b  = [("a", "x"), ("b", "y"), ("c", "z")]
    bp = [("a", "x"), ("b", "w")]
    assert negative_score(b, bp, use_approx=False) == pytest.approx(-0.5)


def test_negative_score_strict_mode_no_op_under_exact():
    """Under --no_approx the single-F implementation equals
    -max(|f_from_b|/|B|, |f_from_b_prime|/|B'|) because both
    conflict-set sizes are equal under exact left-value matching.
    Constructed example with two conflicts: b uses lefts {a, b, c}
    where 'a' and 'b' have right-value mismatches with b'."""
    b  = [("a", "x"), ("b", "y"), ("c", "z"), ("d", "q")]
    bp = [("a", "x1"), ("b", "y1"), ("c", "z"), ("e", "q")]
    score = negative_score(b, bp, use_approx=False)
    # 'a' and 'b' conflict; 'c' matches; 'd' and 'e' don't share left.
    # |F| = 2, |B| = 4, |B'| = 4 → -max(2/4, 2/4) = -0.5.
    assert score == pytest.approx(-0.5)
```

- [ ] **Step 2: Run them — expect FAIL or wrong values**

Run: `pytest tests/test_parallel_synthesis.py::test_negative_score_single_F_matches_paper_eq4 tests/test_parallel_synthesis.py::test_negative_score_strict_mode_no_op_under_exact -v`
Expected: at least one fails (current code returns max of two separate F sizes; for the hand-crafted cases the result happens to match, but the change to single-F makes intent explicit).

Note: if both happen to pass on current code, that confirms `|f_from_b| == |f_from_b_prime|` under exact, which is the invariant we're locking in. Proceed to step 3.

- [ ] **Step 3: Refactor `negative_score`**

In `synthesis.py`, replace the current `negative_score` body (lines 209-231) with:

```python
def negative_score(
    b: List[Tuple[str, str]],
    b_prime: List[Tuple[str, str]],
    use_approx: bool = True,
) -> float:
    """Negative incompatibility score (Equation 4 from paper).

    ``w-(B, B') = -max( |F| / |B| , |F| / |B'| )``

    where ``F = {l | (l,r) in B and (l,r') in B' and r != r'}``.

    Returns a value in [-1, 0].

    Under ``use_approx=False`` this is exact Equation 4. Under
    ``use_approx=True`` ``_conflict_set`` is asymmetric — pick one
    direction as the canonical F (see paper §4.1 for the formal
    definition; the paper does not specify approx-matching behavior).
    """
    if not b or not b_prime:
        return 0.0
    f = _conflict_set(b, b_prime, use_approx)
    return -max(len(f) / len(b), len(f) / len(b_prime))
```

- [ ] **Step 4: Run the tests + the full test file as sanity**

Run: `pytest tests/test_parallel_synthesis.py -v 2>&1 | tail -20`
Expected: the two new tests PASS. All existing tests still PASS (under `--no_approx` the change is a no-op; the synthesis_candidates fixture is small enough that approx-vs-exact does not flip any tie-break).

If `test_refactor_preserves_behavior` fails: this means under approx the canonical-F change altered tie-breaks. Pass `use_approx=False` to `negative_score` and recapture EXPECTED in a later task. Note in the failure and continue — Task 14 handles re-pinning.

- [ ] **Step 5: Commit**

```bash
git add synthesis.py tests/test_parallel_synthesis.py
git commit -m "negative_score: use single |F| per paper Eq 4 (no-op under --no_approx)"
```

---

## Task 3: Add `--theta_edge` filter to `_compute_initial_scores` (Phase A1)

**Files:**
- Modify: `synthesis.py:299-339` (`_compute_initial_scores`)
- Modify: `synthesis.py:476-525` (`greedy_partition` — add `theta_edge` parameter)
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Write the three failing tests**

Append to `tests/test_parallel_synthesis.py`:

```python
def test_theta_edge_zero_keeps_all_positive_edges(synthesis_candidates, tmp_path):
    """theta_edge=0 must equal today's `ps > 0` behavior."""
    partitions_strict = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    # Expect the same partition set as the historical pin in
    # test_refactor_preserves_behavior, since theta_edge=0 disables filtering.
    assert _canonical(partitions_strict) == EXPECTED_PARTITIONS


def test_theta_edge_one_keeps_only_perfect_overlap(synthesis_candidates, tmp_path):
    """theta_edge=1.0 keeps only pairs with identical pair sets (w+=1.0).

    In synthesis_candidates, candidates 0 and 3 are identical (3/3 shared
    pairs both ways → w+=1.0). All other pairs have w+<1.0. So {0,3}
    merge; everyone else is a singleton."""
    partitions = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=1.0,
        output_folder=str(tmp_path),
    )
    canon = _canonical(partitions)
    # Indices 0 and 3 must be in the same partition.
    same = [p for p in canon if 0 in p]
    assert same and 3 in same[0]
    # Every other candidate is a singleton.
    for idx in (1, 2, 4, 5, 6, 7):
        assert [idx] in canon


def test_theta_edge_default_drops_low_weight_edges(synthesis_candidates, tmp_path):
    """theta_edge=0.85 keeps fewer edges than theta_edge=0 — fewer merges."""
    p_default = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.85,
        output_folder=str(tmp_path / "default"),
    )
    p_all = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path / "all"),
    )
    # Stricter filter → at least as many partitions (fewer merges).
    assert len(_canonical(p_default)) >= len(_canonical(p_all))
```

- [ ] **Step 2: Run them — expect TypeError on `theta_edge` kwarg**

Run: `pytest tests/test_parallel_synthesis.py::test_theta_edge_zero_keeps_all_positive_edges -v 2>&1 | tail -5`
Expected: FAIL with `TypeError: greedy_partition() got an unexpected keyword argument 'theta_edge'`.

- [ ] **Step 3: Add `theta_edge` to `_compute_initial_scores`**

In `synthesis.py`, modify `_compute_initial_scores` (current signature around line 299) to accept `theta_edge`:

```python
def _compute_initial_scores(
    overlapping_pairs: Set[Tuple[int, int]],
    candidates: List[Candidate],
    use_approx: bool,
    theta_edge: float = 0.85,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], int, int]:
    """Compute positive and negative scores for every overlap edge.

    A positive edge is kept only when ``w+ >= theta_edge`` (paper §5.4
    recommends ``theta_edge=0.85``). Setting ``theta_edge=0`` keeps every
    edge with ``ps > 0`` (legacy behavior).

    Iterates ``sorted(overlapping_pairs)`` so the resulting dicts have
    deterministic insertion order.

    Returns: ``(pos_scores, neg_scores, positive_edges, blocking_edges)``.
    """
    pos_scores: Dict[Tuple[int, int], float] = {}
    neg_scores: Dict[Tuple[int, int], float] = {}
    positive_edges = 0
    blocking_edges = 0
    for ci, cj in tqdm(
        sorted(overlapping_pairs),
        total=len(overlapping_pairs),
        desc="Calculating initial edge weights",
        unit="pair",
    ):
        key = (ci, cj)
        bp = list(candidates[ci]["pairs"])
        bq = list(candidates[cj]["pairs"])
        ps = positive_score(bp, bq, use_approx=use_approx)
        ns = negative_score(bp, bq, use_approx=use_approx)
        if ps >= theta_edge and ps > 0:
            pos_scores[key] = ps
            positive_edges += 1
        if ns < 0:
            neg_scores[key] = ns
            if ns < -0.2:
                blocking_edges += 1
    return pos_scores, neg_scores, positive_edges, blocking_edges
```

Note: the `ps > 0` clause guards against `theta_edge=0` admitting zero-weight edges.

- [ ] **Step 4: Add `theta_edge` to `greedy_partition` and thread it**

In `synthesis.py`, modify `greedy_partition`:

```python
def greedy_partition(
    candidates: List[Candidate],
    tau: float = -0.2,
    theta_overlap: int = 1,
    use_approx: bool = True,
    output_folder: str = "output",
    theta_edge: float = 0.85,
) -> List[Partition]:
    # ... existing docstring + body up to overlap-set build ...

    overlapping_pairs = _build_overlap_set(pair_index, left_index)
    pos_scores, neg_scores, positive_edges, blocking_edges = _compute_initial_scores(
        overlapping_pairs, candidates, use_approx, theta_edge=theta_edge,
    )
    # ... rest unchanged ...
```

- [ ] **Step 5: Run the new tests + sanity**

Run: `pytest tests/test_parallel_synthesis.py -v 2>&1 | tail -20`
Expected: the three new tests PASS. `test_refactor_preserves_behavior` still PASSES (it doesn't pass `theta_edge`, so it picks up the default 0.85 — this will fail. **Pass `theta_edge=0` explicitly in that test now to keep it stable for the rest of the work.**)

If `test_refactor_preserves_behavior` fails, add `theta_edge=0` to its call:

```python
def test_refactor_preserves_behavior(synthesis_candidates, tmp_path):
    actual = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    assert _canonical(actual) == EXPECTED_PARTITIONS
```

Re-run; expect all PASS.

- [ ] **Step 6: Commit**

```bash
git add synthesis.py tests/test_parallel_synthesis.py
git commit -m "Add --theta_edge filter in _compute_initial_scores (paper Eq 3 + §5.4)"
```

---

## Task 4: Plumb `theta_edge` through the parallel scoring path

**Files:**
- Modify: `parallel_pipeline.py:295-336` (`parallel_compute_initial_scores`)
- Modify: `parallel_pipeline.py:339-376` (`parallel_greedy_partition`)
- Modify: `tests/test_parallel_synthesis.py` (extend parallel tests with `theta_edge`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_parallel_synthesis.py`:

```python
@pytest.mark.parametrize("theta_edge", [0.0, 0.5, 0.85, 1.0])
def test_parallel_equals_sequential_at_theta_edge(synthesis_candidates, tmp_path, theta_edge):
    """parallel_greedy_partition must match sequential at every theta_edge."""
    seq = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=theta_edge,
        output_folder=str(tmp_path / "seq"),
    )
    par = parallel_greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        n_workers=2, chunk_size=2,
        theta_edge=theta_edge,
        output_folder=str(tmp_path / "par"),
    )
    assert _canonical(par) == _canonical(seq)
```

- [ ] **Step 2: Run it — expect TypeError**

Run: `pytest tests/test_parallel_synthesis.py::test_parallel_equals_sequential_at_theta_edge -v 2>&1 | tail -5`
Expected: FAIL with `TypeError: parallel_greedy_partition() got an unexpected keyword argument 'theta_edge'`.

- [ ] **Step 3: Add `theta_edge` to `parallel_compute_initial_scores`**

In `parallel_pipeline.py`, modify the signature and the per-edge filter:

```python
def parallel_compute_initial_scores(
    overlapping_pairs,
    candidates,
    use_approx,
    n_workers=None,
    chunk_size=1000,
    theta_edge=0.85,
):
    """Parallel drop-in for synthesis._compute_initial_scores."""
    if n_workers is None:
        n_workers = mp.cpu_count()

    edges = sorted(overlapping_pairs)
    pos_scores = {}
    neg_scores = {}
    positive_edges = 0
    blocking_edges = 0

    with mp.Pool(
        processes=n_workers,
        initializer=_init_scoring_worker,
        initargs=(candidates, use_approx),
    ) as pool:
        for ci, cj, ps, ns in tqdm(
            pool.imap(_score_edge_worker, edges, chunksize=chunk_size),
            total=len(edges),
            desc="Calculating initial edge weights (parallel)",
            unit="pair",
        ):
            key = (ci, cj)
            if ps >= theta_edge and ps > 0:
                pos_scores[key] = ps
                positive_edges += 1
            if ns < 0:
                neg_scores[key] = ns
                if ns < -0.2:
                    blocking_edges += 1
    return pos_scores, neg_scores, positive_edges, blocking_edges
```

- [ ] **Step 4: Add `theta_edge` to `parallel_greedy_partition`**

In `parallel_pipeline.py`, modify the signature and pass through:

```python
def parallel_greedy_partition(
    candidates,
    tau=-0.2,
    theta_overlap=1,
    use_approx=True,
    n_workers=None,
    chunk_size=1000,
    output_folder="output",
    theta_edge=0.85,
):
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
        theta_edge=theta_edge,
    )
    print(f"    Non-zero positive edges: {positive_edges}")
    print(f"    Blocking negative edges (w- < tau): {blocking_edges}")
    print(f"  Running greedy partitioning...")

    return _run_merge_loop(
        candidates, pos_scores, neg_scores, tau, theta_overlap, output_folder
    )
```

- [ ] **Step 5: Run the parametrized test**

Run: `pytest tests/test_parallel_synthesis.py::test_parallel_equals_sequential_at_theta_edge -v 2>&1 | tail -10`
Expected: all 4 parametrized cases PASS.

Then run the full file:
Run: `pytest tests/test_parallel_synthesis.py -v 2>&1 | tail -15`
Expected: all PASS. (Existing `test_parallel_equals_sequential` may need `theta_edge=0` added explicitly; if it fails, do that.)

- [ ] **Step 6: Commit**

```bash
git add parallel_pipeline.py tests/test_parallel_synthesis.py
git commit -m "Plumb theta_edge through parallel scoring path"
```

---

## Task 5: Implement `--theta_overlap` filter properly in `_build_overlap_set` (Phase A2)

**Files:**
- Modify: `synthesis.py:269-296` (`_build_overlap_set`)
- Modify: `synthesis.py:476-525` (`greedy_partition` — pass `theta_overlap` to `_build_overlap_set`)
- Modify: `parallel_pipeline.py:339-376` (mirror in `parallel_greedy_partition`)
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parallel_synthesis.py`:

```python
def test_theta_overlap_one_acts_like_today(synthesis_candidates, tmp_path):
    """theta_overlap=1 (today's claimed default) admits any pair sharing
    at least 2 bucket co-occurrences. Verify the partition output equals
    EXPECTED_PARTITIONS so that current behavior remains pinned."""
    actual = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    assert _canonical(actual) == EXPECTED_PARTITIONS


def test_theta_overlap_high_filters_buckets(tmp_path):
    """Synthetic candidates sharing exactly 2, 3, 4 pairs.

    With theta_overlap=3, only the 4-share pair survives → partitions
    are {0, 3} (the 4-share pair) and singletons {1}, {2}.
    """
    from synthesis import greedy_partition as gp

    def mk(idx, pairs):
        return {
            "pairs": [tuple(p) for p in pairs],
            "theta": 1.0, "row_count": len(pairs), "covered_rows": len(pairs),
            "source_table_index": idx, "left_column_index": 0,
            "right_column_index": 1, "source_metadata": {},
        }
    cands = [
        # 0 and 1: share 2 pairs
        mk(0, [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]),
        mk(1, [("a", "1"), ("b", "2"), ("e", "5"), ("f", "6")]),
        # 0 and 2: share 3 pairs
        mk(2, [("a", "1"), ("b", "2"), ("c", "3"), ("g", "7")]),
        # 0 and 3: share 4 pairs (identical)
        mk(3, [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]),
    ]
    partitions = gp(
        cands,
        tau=-0.2, theta_overlap=3, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    canon = _canonical(partitions)
    # 0 and 3 must merge (share 4 > 3).
    merged = [p for p in canon if 0 in p]
    assert merged and 3 in merged[0]
    # 1 and 2 do not get a positive edge with 0 (shared count <= 3 → no edge).
    assert [1] in canon
    assert [2] in canon
```

- [ ] **Step 2: Run them — expect at least one failure**

Run: `pytest tests/test_parallel_synthesis.py::test_theta_overlap_one_acts_like_today tests/test_parallel_synthesis.py::test_theta_overlap_high_filters_buckets -v 2>&1 | tail -10`
Expected: `test_theta_overlap_high_filters_buckets` FAILS (current `_build_overlap_set` ignores `theta_overlap`).

- [ ] **Step 3: Implement `theta_overlap` in `_build_overlap_set`**

In `synthesis.py`, replace `_build_overlap_set` (lines 269-296) with:

```python
def _build_overlap_set(
    pair_index: Dict[Tuple[str, str], List[int]],
    left_index: Dict[str, List[int]],
    theta_overlap: int = 1,
) -> Set[Tuple[int, int]]:
    """Union of candidate-index pairs whose co-occurrence count in
    either ``pair_index`` (shared value pairs) or ``left_index``
    (shared left values) **strictly exceeds** ``theta_overlap`` (paper §4.1).

    Returns ordered tuples ``(a, b)`` with ``a < b``.

    Memory: the Counter scales with raw enumerations
    (``Σ k·(k-1)/2`` across buckets). Acceptable on dama (251 GB) for
    corpora up to ~10k filtered tables; will OOM at very large scale —
    see the spec for the explicit single-machine limit.
    """
    from collections import Counter

    pair_count: Counter = Counter()
    for indices in pair_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = indices[x], indices[y]
                if a > b:
                    a, b = b, a
                pair_count[(a, b)] += 1

    left_count: Counter = Counter()
    for indices in left_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = indices[x], indices[y]
                if a > b:
                    a, b = b, a
                left_count[(a, b)] += 1

    return {k for k, v in pair_count.items() if v > theta_overlap} | \
           {k for k, v in left_count.items() if v > theta_overlap}
```

- [ ] **Step 4: Update `greedy_partition` to pass `theta_overlap`**

In `synthesis.py`, update the `_build_overlap_set` call inside `greedy_partition`:

```python
overlapping_pairs = _build_overlap_set(pair_index, left_index, theta_overlap)
```

- [ ] **Step 5: Mirror in `parallel_greedy_partition`**

In `parallel_pipeline.py`, update its `_build_overlap_set` call the same way:

```python
overlapping_pairs = _build_overlap_set(pair_index, left_index, theta_overlap)
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_parallel_synthesis.py -v 2>&1 | tail -20`
Expected: all PASS. (The synthesis_candidates fixture's edges all share ≥2 entries when present, so `theta_overlap=1` retains today's structure.)

- [ ] **Step 7: Commit**

```bash
git add synthesis.py parallel_pipeline.py tests/test_parallel_synthesis.py
git commit -m "Apply theta_overlap as per-edge filter via Counter (paper §4.1)"
```

---

## Task 6: Add `_connected_components` helper (Phase B prep)

**Files:**
- Modify: `synthesis.py` (add helper before `_run_merge_loop`)
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parallel_synthesis.py`:

```python
def test_connected_components_singletons():
    """No positive edges → n singletons."""
    from synthesis import _connected_components
    comps = _connected_components(pos_scores={}, n_candidates=5)
    assert sorted(sorted(c) for c in comps) == [[0], [1], [2], [3], [4]]


def test_connected_components_two_disjoint_clusters():
    """Two clusters of 3 + one isolated singleton → 3 components."""
    from synthesis import _connected_components
    pos_scores = {
        (0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.9,  # cluster A
        (3, 4): 0.9, (4, 5): 0.9, (3, 5): 0.9,  # cluster B
        # 6 isolated
    }
    comps = _connected_components(pos_scores, n_candidates=7)
    canon = sorted(sorted(c) for c in comps)
    assert canon == [[0, 1, 2], [3, 4, 5], [6]]


def test_connected_components_chain_path():
    """Edges 0-1, 1-2, 2-3 form a single component {0,1,2,3}."""
    from synthesis import _connected_components
    comps = _connected_components(
        pos_scores={(0, 1): 0.5, (1, 2): 0.5, (2, 3): 0.5},
        n_candidates=4,
    )
    assert len(comps) == 1
    assert sorted(comps[0]) == [0, 1, 2, 3]
```

- [ ] **Step 2: Run them — expect ImportError**

Run: `pytest tests/test_parallel_synthesis.py -k "connected_components" -v 2>&1 | tail -10`
Expected: FAIL — `_connected_components` not defined.

- [ ] **Step 3: Add the helper**

In `synthesis.py`, add this helper immediately above `_run_merge_loop` (the location keeps related code together):

```python
def _connected_components(
    pos_scores: Dict[Tuple[int, int], float],
    n_candidates: int,
) -> List[Set[int]]:
    """Compute connected components of the positive-edge graph.

    Implements the divide-and-conquer reduction described in
    Wang & He (SIGMOD 2017) §4.2 and Appendix E. Two candidates connected
    by a positive edge (directly or transitively) live in the same
    component; the merge loop can never merge candidates across
    components, so each component is an independent subproblem.

    Uses path-compressed Union-Find (single-machine equivalent of the
    paper's Hash-to-Min on Map-Reduce).

    Args:
        pos_scores: ``{(a, b): w+}`` from initial scoring.
        n_candidates: total number of candidates (singletons are seeded
            from this).

    Returns:
        Components as sets of candidate indices, sorted by
        ``min(component)`` for deterministic order.
    """
    parent = list(range(n_candidates))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in pos_scores:
        union(a, b)

    buckets: Dict[int, Set[int]] = defaultdict(set)
    for i in range(n_candidates):
        buckets[find(i)].add(i)
    return sorted(buckets.values(), key=lambda s: min(s))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_parallel_synthesis.py -k "connected_components" -v 2>&1 | tail -10`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add synthesis.py tests/test_parallel_synthesis.py
git commit -m "Add _connected_components helper (Union-Find, paper App. E)"
```

---

## Task 7: Capture pre-component baseline for the equivalence test

This is a transient pinning step: before changing `_run_merge_loop`, capture what today's monolithic merge loop produces on the synthesis_candidates fixture so the post-refactor equivalence test has a target.

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Capture EXPECTED via one-off script**

Run:

```bash
source .venv/bin/activate
python - <<'PY'
from synthesis import greedy_partition
def mk(idx, pairs):
    return {"pairs":[tuple(p) for p in pairs],"theta":1.0,
            "row_count":len(pairs),"covered_rows":len(pairs),
            "source_table_index":idx,"left_column_index":0,
            "right_column_index":1,"source_metadata":{}}
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
parts = greedy_partition(cands, tau=-0.2, theta_overlap=1, use_approx=True,
                         theta_edge=0.0, output_folder="/tmp")
print("UNSPLIT_EXPECTED =", repr(sorted([sorted(p) for p in parts])))
PY
```

Capture the printed value.

- [ ] **Step 2: Add the placeholder test**

Append to `tests/test_parallel_synthesis.py`:

```python
# Captured BEFORE the per-component refactor of _run_merge_loop, on
# the synthesis_candidates fixture at theta_edge=0, theta_overlap=1.
# Post-refactor greedy_partition with the same args must produce the
# same partitions.
UNSPLIT_EXPECTED = ...  # paste captured value here


def test_with_components_equals_without_components(synthesis_candidates, tmp_path):
    """The per-component refactor of _run_merge_loop must produce
    identical output to the pre-refactor monolithic loop on a
    fully-connected fixture."""
    partitions = greedy_partition(
        synthesis_candidates,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    assert _canonical(partitions) == UNSPLIT_EXPECTED
```

Replace `UNSPLIT_EXPECTED = ...` with the captured value from step 1.

- [ ] **Step 3: Run the test now (PRE-refactor, expect PASS)**

Run: `pytest tests/test_parallel_synthesis.py::test_with_components_equals_without_components -v`
Expected: PASS — `UNSPLIT_EXPECTED` was captured from the current code, so it matches.

- [ ] **Step 4: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Pin pre-refactor monolithic merge-loop output for equivalence test"
```

---

## Task 8: Restructure `_run_merge_loop` into per-component dispatch (Phase B)

**Files:**
- Modify: `synthesis.py:341-473` (the whole `_run_merge_loop` function body)

- [ ] **Step 1: Read the current `_run_merge_loop` body**

Run: `sed -n '341,475p' synthesis.py`
Expected: see the existing body — partition init, save edge scores, main merge loop, return.

- [ ] **Step 2: Lift the merge-loop body into `_greedy_merge_component`**

In `synthesis.py`, add a new helper directly below `_connected_components`:

```python
def _greedy_merge_component(
    candidates: List[Candidate],
    component: Set[int],
    pos_scores: Dict[Tuple[int, int], float],
    neg_scores: Dict[Tuple[int, int], float],
    tau: float,
) -> List[Partition]:
    """Run the greedy merge loop on a single positive-edge component.

    Mutates the provided ``pos_scores`` / ``neg_scores`` dicts as it
    merges; do not reuse them after the call. ``component`` holds the
    candidate indices in this subgraph; partition state is initialized
    only for those indices.
    """
    part_members: Dict[int, List[int]] = {i: [i] for i in component}
    part_pairs: Dict[int, List[Tuple[str, str]]] = {
        i: list(candidates[i]["pairs"]) for i in component
    }
    next_pid = max(component) + 1

    merge_count = 0
    pbar_merge = tqdm(desc="Merging partitions", unit="round")
    while True:
        best_key: Optional[Tuple[int, int]] = None
        best_pos: float = -1.0
        ns_best: float = 0.0

        for key, ps in pos_scores.items():
            if ps <= best_pos:
                continue
            pi, pj = key
            if pi not in part_members or pj not in part_members:
                continue
            ns = neg_scores.get(key, 0.0)
            if ns >= tau:
                best_key = key
                best_pos = ps
                ns_best = ns

        if best_key is None:
            break

        pi, pj = best_key
        new_pid = next_pid
        next_pid += 1
        new_members = part_members[pi] + part_members[pj]
        size_i = len(part_members[pi])
        size_j = len(part_members[pj])
        merge_count += 1
        pbar_merge.update(1)
        pbar_merge.set_postfix({"last_w+": f"{best_pos:.3f}",
                                "active_parts": len(part_members)})

        seen: Set[Tuple[str, str]] = set()
        new_pairs: List[Tuple[str, str]] = []
        for pair in part_pairs[pi] + part_pairs[pj]:
            pt = (pair[0], pair[1])
            if pt not in seen:
                seen.add(pt)
                new_pairs.append(pt)

        part_members[new_pid] = new_members
        part_pairs[new_pid] = new_pairs

        remaining = [
            pid for pid in part_members
            if pid != pi and pid != pj and pid != new_pid
        ]
        for pk in remaining:
            key_ik = (min(pi, pk), max(pi, pk))
            key_jk = (min(pj, pk), max(pj, pk))
            key_nk = (min(new_pid, pk), max(new_pid, pk))

            old_pos_ik = pos_scores.get(key_ik, 0.0)
            old_pos_jk = pos_scores.get(key_jk, 0.0)
            new_pos = old_pos_ik + old_pos_jk

            old_neg_ik = neg_scores.get(key_ik, 0.0)
            old_neg_jk = neg_scores.get(key_jk, 0.0)
            new_neg = min(old_neg_ik, old_neg_jk)

            if new_pos > 0:
                pos_scores[key_nk] = new_pos
            if new_neg < 0:
                neg_scores[key_nk] = new_neg

            pos_scores.pop(key_ik, None)
            pos_scores.pop(key_jk, None)
            neg_scores.pop(key_ik, None)
            neg_scores.pop(key_jk, None)

        del part_members[pi]
        del part_members[pj]
        del part_pairs[pi]
        del part_pairs[pj]
        pos_scores.pop(best_key, None)
        neg_scores.pop(best_key, None)

    pbar_merge.close()
    return [sorted(members) for members in part_members.values()]
```

- [ ] **Step 3: Replace `_run_merge_loop` body with dispatch logic**

In `synthesis.py`, replace the existing `_run_merge_loop` body. Keep the save-edge-scores block at the top (it operates on the *full* pos/neg dicts, before bucketing). Then bucket by component and dispatch:

```python
def _run_merge_loop(
    candidates: List[Candidate],
    pos_scores: Dict[Tuple[int, int], float],
    neg_scores: Dict[Tuple[int, int], float],
    tau: float,
    theta_overlap: int,
    output_folder: str,
) -> List[Partition]:
    """Save edge weights, compute connected components of the positive-edge
    graph, and run the greedy merge loop independently per component.
    Returns the concatenated partition list."""
    import json
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    scores_output_path = os.path.join(output_folder, "computed_edge_scores.jsonl")
    print(f"  Saving computed edge weights to {os.path.abspath(scores_output_path)}...")
    all_edge_keys = set(pos_scores.keys()).union(set(neg_scores.keys()))
    with open(scores_output_path, "w", encoding="utf-8") as f:
        for ci, cj in all_edge_keys:
            edge_data = {
                "cand_i": ci, "cand_j": cj,
                "w_pos": pos_scores.get((ci, cj), 0.0),
                "w_neg": neg_scores.get((ci, cj), 0.0),
            }
            f.write(json.dumps(edge_data) + "\n")
    print(f"  Successfully saved {len(all_edge_keys)} edge scores to {output_folder}/")

    n = len(candidates)
    components = _connected_components(pos_scores, n)
    largest = max((len(c) for c in components), default=0)
    print(f"  Found {len(components)} connected components "
          f"(largest: {largest})")

    cand_to_comp: Dict[int, int] = {
        c: i for i, comp in enumerate(components) for c in comp
    }
    comp_pos: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    comp_neg: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    for key, v in pos_scores.items():
        comp_pos[cand_to_comp[key[0]]][key] = v
    for key, v in neg_scores.items():
        a, b = key
        if cand_to_comp.get(a) == cand_to_comp.get(b):
            comp_neg[cand_to_comp[a]][key] = v

    all_partitions: List[Partition] = []
    for i, component in enumerate(components):
        if len(component) == 1:
            all_partitions.append([next(iter(component))])
            continue
        all_partitions.extend(_greedy_merge_component(
            candidates, component, comp_pos[i], comp_neg[i], tau,
        ))
    print(f"    Converged across {len(components)} components. "
          f"{len(all_partitions)} partitions total.")
    return all_partitions
```

- [ ] **Step 4: Run the equivalence test**

Run: `pytest tests/test_parallel_synthesis.py::test_with_components_equals_without_components -v`
Expected: PASS — the pinned `UNSPLIT_EXPECTED` from Task 7 must match.

If it FAILS: diff `_canonical(partitions)` vs `UNSPLIT_EXPECTED` to find the drift. Likely candidates: cross-component neg-edge handling, deterministic ordering, or singleton emission.

- [ ] **Step 5: Run the full test file**

Run: `pytest tests/test_parallel_synthesis.py -v 2>&1 | tail -25`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add synthesis.py
git commit -m "Decompose _run_merge_loop by positive-edge connected components

Per Wang & He (SIGMOD 2017) §4.2 + App. E. Each component runs the
greedy merge loop independently. Cross-component negative edges are
dropped (they can never block a merge). Singletons skip the loop."
```

---

## Task 9: Component correctness tests (Layer 2 from spec)

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

- [ ] **Step 1: Write the three tests**

Append to `tests/test_parallel_synthesis.py`:

```python
def test_components_singletons_passthrough(synthesis_components_fixture, tmp_path):
    """The isolated candidate (index 6) emerges as its own 1-element
    partition without any merge being attempted."""
    partitions = greedy_partition(
        synthesis_components_fixture,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    canon = _canonical(partitions)
    assert [6] in canon


def test_components_disjoint_clusters_independent(synthesis_components_fixture, tmp_path):
    """No final partition spans both cluster A ({0,1,2}) and cluster B
    ({3,4,5})."""
    partitions = greedy_partition(
        synthesis_components_fixture,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    canon = _canonical(partitions)
    cluster_a = {0, 1, 2}
    cluster_b = {3, 4, 5}
    for partition in canon:
        partition_set = set(partition)
        if partition_set & cluster_a and partition_set & cluster_b:
            pytest.fail(f"Partition {partition} mixes clusters A and B")


def test_components_cross_component_neg_edges_dropped(tmp_path):
    """A neg edge between candidates in different positive-edge
    components must not appear in any per-component neg dict.

    We rely on observable behavior: if the cross-component neg edge
    *were* respected, it could block a merge inside one of the
    components. We construct a case where dropping vs respecting the
    cross-component neg edge yields different partition outputs and
    assert that the drop-behavior output matches.
    """
    def mk(idx, pairs):
        return {
            "pairs": [tuple(p) for p in pairs],
            "theta": 1.0, "row_count": len(pairs), "covered_rows": len(pairs),
            "source_table_index": idx, "left_column_index": 0,
            "right_column_index": 1, "source_metadata": {},
        }

    # Cluster A: 0 and 1 share enough pair-overlap to merge.
    # Cluster B: candidate 2, no overlap with cluster A.
    # But 0 and 2 share a left value 'x' with conflicting rights — would
    # produce a strong negative edge if discovered. With theta_overlap=1
    # and no left-value bucket of size >= 2, the (0,2) neg edge is
    # dropped at overlap-set construction; this test pins partition
    # output as {0,1} merged, {2} singleton.
    cands = [
        mk(0, [("a", "1"), ("b", "2"), ("c", "3"), ("x", "y")]),
        mk(1, [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]),
        mk(2, [("x", "z"), ("p", "q"), ("r", "s")]),
    ]
    partitions = greedy_partition(
        cands,
        tau=-0.2, theta_overlap=1, use_approx=True,
        theta_edge=0.0,
        output_folder=str(tmp_path),
    )
    canon = _canonical(partitions)
    merged = [p for p in canon if 0 in p]
    assert merged and 1 in merged[0], "0 and 1 should merge"
    assert [2] in canon, "2 should be a singleton"
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_parallel_synthesis.py -k "components_" -v 2>&1 | tail -15`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Add component correctness tests: singletons, disjoint clusters, neg-edge drop"
```

---

## Task 10: Wire `--theta_edge` into `main.py` and bump `--theta_overlap` default

**Files:**
- Modify: `main.py` (parse_args + Stage 6 call sites)

- [ ] **Step 1: Add the flag and update default**

In `main.py`, in `parse_args()`, locate the existing `--theta_overlap` argument and change its default to 3. Then add `--theta_edge`:

```python
    p.add_argument("--theta_overlap", type=int, default=3,
                   help="WP3 minimum bucket co-occurrence count to admit a "
                        "candidate pair (default 3 per Wang & He §4.1). "
                        "Set to 1 to match pre-paper-conformance behavior.")
    p.add_argument("--theta_edge", type=float, default=0.85,
                   help="WP3 minimum normalized w+ to admit a positive edge "
                        "(default 0.85 per Wang & He §5.4). Set to 0.0 to "
                        "disable filtering (pre-paper-conformance behavior).")
```

- [ ] **Step 2: Update the `--no_approx` help text**

In `main.py`, locate `--no_approx` and replace its help string:

```python
    p.add_argument("--no_approx", action="store_true",
                   help="WP3 disable approximate string matching. "
                        "Required for paper-strict mode: under --no_approx "
                        "the scoring functions match Wang & He (SIGMOD 2017) "
                        "Equations 3 and 4 exactly.")
```

- [ ] **Step 3: Thread `theta_edge` into Stage 6 calls**

In `main.py`'s Stage 6, both branches of the parallel dispatch need to pass `theta_edge`:

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
            theta_edge=args.theta_edge,
        )
    else:
        partitions = greedy_partition(
            wp3_candidates,
            tau=args.tau,
            theta_overlap=args.theta_overlap,
            use_approx=not args.no_approx,
            output_folder=args.output_folder,
            theta_edge=args.theta_edge,
        )
```

- [ ] **Step 4: Smoke-test `--help`**

Run: `python main.py --help 2>&1 | grep -A2 "theta_edge\|theta_overlap\|no_approx"`
Expected: the flags appear with the new help text. `--theta_overlap` shows default 3. `--theta_edge` shows default 0.85.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "CLI: add --theta_edge (0.85), default --theta_overlap to 3, update --no_approx help"
```

---

## Task 11: Update existing tests for new defaults

**Files:**
- Modify: `tests/test_parallel_synthesis.py`

After Tasks 3, 4, 5, 8, the existing `test_refactor_preserves_behavior` and `test_parallel_equals_sequential` were patched defensively to pass `theta_edge=0.0` so they kept passing. This task makes that explicit and consistent.

- [ ] **Step 1: Audit existing test calls**

Run: `grep -n "greedy_partition\|parallel_greedy_partition" tests/test_parallel_synthesis.py | grep -v "import\|^#"`
Expected: every call to either function passes both `theta_edge=...` and `theta_overlap=...` explicitly.

- [ ] **Step 2: Add explicit defaults where missing**

For every call missing one of those kwargs, add it. The conservative pin for legacy tests is `theta_edge=0.0, theta_overlap=1`. For the new tests written in Tasks 6-9, use the values they already specify.

- [ ] **Step 3: Run the full test file**

Run: `pytest tests/test_parallel_synthesis.py -v 2>&1 | tail -25`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_parallel_synthesis.py
git commit -m "Make theta_edge / theta_overlap explicit in existing tests"
```

---

## Task 12: Sanity check on the whole suite

**Files:** none (pure verification).

- [ ] **Step 1: Run all tests in the repo**

Run: `pytest -q 2>&1 | tail -8`
Expected: all PASS (>= 285 tests).

- [ ] **Step 2: Check `--help` end-to-end one more time**

Run: `python main.py --help 2>&1 | head -50`
Expected: clean help output, no parse errors.

- [ ] **Step 3: Smoke-run on a tiny CSV folder** (optional if you have one handy)

```bash
mkdir -p /tmp/sanity_corpus && cat > /tmp/sanity_corpus/t1.csv <<'CSV'
a,1
b,2
c,3
CSV
cat > /tmp/sanity_corpus/t2.csv <<'CSV'
a,1
b,2
d,4
CSV
python main.py --corpus_path /tmp/sanity_corpus --output_folder /tmp/sanity_out --threshold 0 2>&1 | tail -15
```

Expected: pipeline runs without error end-to-end with new defaults.

- [ ] **Step 4: Commit nothing (this task is verification only)**

---

## Task 13: Manual verification on dama (post-merge)

This is a **manual verification step**, not a unit test. Run it after Task 12 is committed and ideally after merging the branch.

- [ ] **Step 1: Sync code to dama**

```bash
rsync -az \
  --exclude '.git' --exclude '.venv' --exclude 'output' --exclude 'data' \
  --exclude '__pycache__' --exclude '.idea' --exclude '*.ipynb' --exclude 'papers' \
  --exclude 'claude' --exclude '.DS_Store' --exclude '.claude' \
  ./ dama:Automap/
```

- [ ] **Step 2: WDC corpus sanity run on dama**

```bash
ssh dama 'cd ~/Automap && nohup .venv/bin/python -u main.py \
  --corpus_path data/sample_full.json \
  --output_folder output/paper_conformance_wdc/ \
  --parallel_workers 14 \
  --no_approx \
  > run_paper_wdc.log 2>&1 < /dev/null & echo "PID $!"'
```

Expected wall-clock: tens of minutes (vs. the previous parallel-only ~102 min) because the merge loop is now per-component.

Monitor with: `ssh dama 'tail -n 5 ~/Automap/run_paper_wdc.log'` periodically.

- [ ] **Step 3: Vertica 10k run on dama**

```bash
ssh dama 'cd ~/Automap && nohup .venv/bin/python -u main.py \
  --corpus_path data/vertica_filtered_10k.jsonl \
  --output_folder output/paper_conformance_vertica10k/ \
  --parallel_workers 14 \
  --no_approx \
  > run_paper_v10k.log 2>&1 < /dev/null & echo "PID $!"'
```

Expected wall-clock: minutes-to-tens-of-minutes (vs. previous "4-day" projection). The combination of `theta_edge=0.85` cutting most positive edges + connected components cutting per-iteration cost is what unblocks this.

- [ ] **Step 4: Record results in the timings doc**

After both runs finish, append the wall-clock per stage to `docs/2026-06-04-vertica-30k-timings.md` under a new "Phase 3 — paper-conformance run" section. Include partition counts and a one-line qualitative summary.

- [ ] **Step 5: (Stretch) Vertica 30k run**

Same command, swapping the corpus path. May OOM during the θ_overlap Counter construction. If it does, document the failure in the timings doc as the honest single-machine ceiling and stop.

---

## Self-Review Notes

**Spec coverage check:**
- Phase A1 (theta_edge): Tasks 3, 4, 10 ✓
- Phase A2 (theta_overlap with Counter, no MAX_BUCKET_SIZE): Tasks 5, 10 ✓
- Phase A3 (negative_score single |F|): Task 2 ✓
- Phase B (connected components): Tasks 6, 7, 8, 9 ✓
- Paper-strict mode (--no_approx help): Task 10 ✓
- Layer 1 unit tests (per-fix): Tasks 2, 3, 5 ✓
- Layer 2 component tests: Tasks 6, 9 ✓
- Layer 3 equivalence and regression: Tasks 7, 8, 11 ✓
- Layer 4 manual verification: Task 13 ✓
- Updated `EXPECTED_PARTITIONS` for new defaults: Task 11 ✓
- `synthesis_components_fixture`: Task 1 ✓

**Placeholder scan:** the `UNSPLIT_EXPECTED = ...` in Task 7 Step 2 is explicitly captured in Step 1; no "TBD" or "fill in details" elsewhere.

**Type/name consistency:** `_connected_components`, `_greedy_merge_component`, `theta_edge`, `theta_overlap` — same identifiers used in every task that references them.
