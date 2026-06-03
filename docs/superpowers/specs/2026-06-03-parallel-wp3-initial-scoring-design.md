# Parallel WP3 Initial Scoring — Design

**Date:** 2026-06-03
**Status:** Draft, awaiting approval

## Goal

`greedy_partition` (WP3 / `synthesis.py`) does not scale. On the WDC corpus
(33,283 candidates) it took 25,701 s (~7.14 h) on dama, almost entirely spent
in **Step 2** — building the initial compatibility graph by computing
`positive_score` and `negative_score` for every overlap edge. Step 2 is
embarrassingly parallel (each edge is scored independently). Parallelizing
it is the single highest-leverage change available without altering the
algorithm.

This document specifies that parallelization. Step 3 (the greedy merge loop)
remains sequential — its optimization (priority queue / heap) is out of scope
here and tracked as future work.

## Non-goals

- Optimizing or parallelizing the greedy merge loop (Step 3).
- Changing the algorithm's output. Parallel must produce **bit-identical**
  partitions to sequential (see Determinism Contract).
- Changing `greedy_partition`'s public signature or behavior. Existing callers
  are unaffected.
- Wiring parallelism into WP1/WP2 stages — already done in
  `parallel_pipeline.parallel_score_corpus` / `parallel_fd_filter`.

## Architecture

### Refactor in `synthesis.py`

Extract three private helpers from inside `greedy_partition`. The function
becomes pure orchestration:

```python
def greedy_partition(candidates, tau=-0.2, theta_overlap=1, use_approx=True,
                     output_folder="output") -> List[Partition]:
    pair_index, left_index = build_inverted_index(candidates)
    overlapping_pairs = _build_overlap_set(pair_index, left_index)
    pos_scores, neg_scores = _compute_initial_scores(
        overlapping_pairs, candidates, use_approx
    )
    return _run_merge_loop(
        candidates, pos_scores, neg_scores, tau, theta_overlap, output_folder
    )
```

- `_build_overlap_set(pair_index, left_index) -> Set[Tuple[int, int]]` —
  wraps today's lines ~318-336 (the two nested loops over the inverted
  indexes that union candidate-pair overlap edges).
- `_compute_initial_scores(overlapping_pairs, candidates, use_approx)
  -> (pos_scores, neg_scores)` — wraps today's lines ~345-361.
- `_run_merge_loop(candidates, pos_scores, neg_scores, tau, theta_overlap,
  output_folder) -> List[Partition]` — wraps today's lines ~366-449,
  including the `computed_edge_scores.jsonl` dump zeynep added.

This refactor changes no behavior; a regression test pins the existing
sequential output to detect drift.

### Additions in `parallel_pipeline.py`

- `_CANDIDATES`, `_USE_APPROX` — module-level globals populated in worker
  processes by `_init_scoring_worker`.
- `_init_scoring_worker(candidates, use_approx)` — top-level `Pool`
  initializer; assigns the globals.
- `_score_edge_worker(edge: Tuple[int, int]) -> Tuple[int, int, float, float]`
  — top-level worker. Returns `(ci, cj, pos, neg)`.
- `parallel_compute_initial_scores(overlapping_pairs, candidates, use_approx,
  n_workers=None, chunk_size=1000) -> (pos_scores, neg_scores)` — drop-in
  replacement for the sequential helper.
- `parallel_greedy_partition(candidates, tau=-0.2, theta_overlap=1,
  use_approx=True, n_workers=None, chunk_size=1000,
  output_folder="output") -> List[Partition]` — same orchestration as
  `greedy_partition`, but uses the parallel scorer. `output_folder` is
  threaded through to `_run_merge_loop` so the `computed_edge_scores.jsonl`
  dump is produced in both paths.
- Optional: `benchmark_synthesis(candidates, n_workers_list=None)` mirroring
  the existing `benchmark()` for after-deployment validation.

## Worker model & data flow

**Pool setup (parent):**

```python
edges = sorted(overlapping_pairs)               # deterministic order
with mp.Pool(
    n_workers,
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
        if ps > 0:
            pos_scores[key] = ps
        if ns < 0:
            neg_scores[key] = ns
```

**Per task:** parent sends only `(ci, cj)` — ~16 bytes. Worker looks up
candidates from its module-level global.

**Memory budget (per worker):** `candidates` is pickled once at worker
startup. For 33k WDC candidates ≈ 20 MB / worker × 28 workers = ~560 MB.
For 1M candidates ≈ 600 MB / worker × 24 workers ≈ 14 GB. Both comfortable
on dama (251 GB).

**Chunk size default:** 1000 edges per task (vs. `parallel_pipeline`'s 50
for tables — edges are cheaper per unit).

## Determinism contract

`parallel_greedy_partition(c, n_workers=N)` returns **the same partition
list** as `greedy_partition(c)` for any input. Three risks neutralized:

1. **Set iteration order.** `overlapping_pairs` is a `set`. Sequential
   iterates set-order; if parallel iterates sorted-order, the resulting
   `pos_scores` dicts have different insertion order. The merge loop scans
   `pos_scores.items()` to find max — on **score ties** the two paths could
   pick different merges → divergent partitions.

   **Fix:** the extracted `_compute_initial_scores` *also* iterates
   `sorted(overlapping_pairs)`. Both paths share the same deterministic
   order — no tie-break drift.

2. **`pool.imap` result order.** `imap` returns results in submission order;
   free given (1).

3. **Floating-point reproducibility.** `positive_score` / `negative_score`
   are pure functions of their inputs; no parallel-reduction summing. Scores
   are bit-identical, not within `1e-6`.

## `main.py` integration

Add one CLI flag:

```
--parallel_workers N    Default 1 (sequential).  >1 routes WP3 through
                        parallel_greedy_partition.
```

Stage 6 of `main.py`:

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
        wp3_candidates, tau=..., theta_overlap=..., use_approx=...,
        output_folder=args.output_folder,
    )
```

Why explicit flag rather than auto-detect: WP3 cost is driven by overlap-edge
count, not candidate count, and overlap is only known after building the
inverted index. An auto-decision would have to live inside `greedy_partition`,
complicating the API.

`computed_edge_scores.jsonl` (zeynep's addition) is written in both paths;
behavior is preserved.

## Recommended worker counts

dama is 14 physical cores × 2 SMT threads = 28 logical.

| Setting | When |
|---|---|
| `--parallel_workers 14` | **Default** on dama — one process per physical core, max efficiency-per-watt, headroom for other users. |
| `--parallel_workers 24` | When the box is ours and we want max throughput. |
| `--parallel_workers 28` | Avoid on shared box. |
| `--parallel_workers 6` | Laptop (M1 Pro: 8 P-cores, leave 2 for system). |

## Testing

New file `tests/test_parallel_synthesis.py`.

| # | Test | Guards |
|---|---|---|
| 1 | `test_parallel_equals_sequential` | The critical one. Hand-crafted 8-candidate fixture exercising identity merge, blocked-by-tau merge, score-tie, no-overlap. Runs both paths at `n_workers ∈ {2, 4}`, asserts `seq == par`. |
| 2 | `test_refactor_preserves_behavior` | Hard-codes expected partitions for the fixture. Future-proofs the synthesis.py extraction. |
| 3 | `test_parallel_n_workers_one` | `n_workers=1` runs through the Pool path correctly. |
| 4 | `test_parallel_edge_cases` | Empty, single candidate, no overlap, all overlap. |
| 5 | `test_parallel_determinism` | Re-run with same input → identical output. |

**Fixture strategy:** hand-craft the candidates in `conftest.py`, no
dependency on WP1/WP2. Keeps the test focused on synthesis.

**Manual verification (post-merge):** run `main.py --parallel_workers 14` on
the existing WDC `candidates.jsonl` on dama, compare partition list to the
sequential run from 2026-05-20 (already on disk in
`/home/automap/Automap/output/synthesized_mappings.jsonl`). Expect:
identical output, wall-clock ~30 min vs. 25,701 s.

## Expected speedup

Step 2 dominates the sequential 25,701 s. With `n_workers=14` on dama:

- Step 2: ~7 hr → ~30 min (linear with physical cores, minus a few % Python
  overhead).
- Steps 1, 3, 4 (build index, merge loop, save): roughly constant at a few
  minutes.
- Total wall-clock: **~30–40 min** for 33k WDC candidates.

At `n_workers=24` the marginal gain from SMT siblings is ~10-15% (not 1.7×);
expected total ~25 min.

## Out of scope / future work

- Step 3 (merge loop) priority-queue optimization. The merge-max scan is
  O(|pos_scores|) per iteration. A heap would make it O(log n). Real win at
  100k+ candidates. Separate spec.
- Within-iteration parallel score recomputation after a merge. Modest gain,
  significant added complexity.
- Algorithm change: lazy negative scoring (computing conflicts only on
  partition-pair evaluation rather than upfront). Bigger architectural
  change; previously deferred.
