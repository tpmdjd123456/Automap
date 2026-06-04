# Paper-Conformance Fixes for WP3 — Design

**Date:** 2026-06-04
**Status:** Draft, awaiting approval

## Goal

The current WP3 synthesis runs at roughly four orders of magnitude beyond the
paper's expected per-iteration cost: the merge loop scans all positive edges
each round (5 s/round on 70 M edges in the 10k Vertica run), and the overlap
discovery materializes O(Σ k²) edges with no per-edge threshold. The paper
(Wang & He, SIGMOD 2017) explicitly describes the scalability techniques that
make their 100M-table experiments feasible. We are missing three of them.

This spec packages all three as a single "paper-conformance" landing:

1. **`--theta_edge` filter** (§5.4 of the paper, default 0.85) — drop
   positive edges with `w⁺ < θ_edge` at scoring time.
2. **`--theta_overlap` properly applied** (§4.1, default change 1 → 3) — only
   evaluate `w(B, B')` if the two candidates share more than `θ_overlap`
   value-pairs (positive) or left-values (negative).
3. **Connected-component decomposition** (§4.2 + Appendix E) — partition the
   candidate set by connected components of the positive-edge graph and run
   the merge loop independently per component.

Plus two strict-fidelity fixes uncovered during audit:

4. **`negative_score` single `|F|`** — current code computes two F sets
   (`f_from_b`, `f_from_b_prime`) and uses them as separate numerators. The
   paper's Eq 4 uses a single `|F|`. Under exact matching this is a no-op;
   under approx it removes an asymmetry.
5. **Paper-strict mode documented** — under `--no_approx` the scoring
   functions are bit-for-bit Equation 3 and Equation 4. Under approx they are
   documented as extensions.

## Non-goals

- Parallelism *across* connected components (serial-first; can layer on
  later, mirroring how parallel scoring was a separate concern from the
  algorithm refactor).
- Union-Find replacement of `part_members` / `part_pairs` inside the merge
  loop. (Union-Find is used between candidates to discover components, but
  partition state inside `_greedy_merge_component` keeps today's
  dict-of-lists.)
- Replacing `_count_intersection`'s asymmetric approx behavior. Under
  `--no_approx` it's exact; under approx it's a documented extension.
- Re-running on the full Vertica corpus. Manual verification on WDC + Vertica
  10k is in scope (see Testing §Layer 4); 30k / 145M is left as a follow-up
  experiment.

## Audit: current code vs. paper

Faithful today:

| Paper element | Code location | Notes |
|---|---|---|
| Equation 3 (`w⁺ = max(intersection/|B|, …)`) | `positive_score` | Exact under `--no_approx`. |
| Equation 4 shape (`w⁻ = -max(…)`) | `negative_score` shape | Numerators are not single `|F|` — see fix below. |
| Algorithm 3 line 14 (`w⁺(P', Pi) ← w⁺(Pi,P1) + w⁺(Pi,P2)`) | merge-loop score aggregation | Exact. |
| Algorithm 3 line 15 (`w⁻(P', Pi) ← min(…)`) | merge-loop score aggregation | Exact. |
| Algorithm 3 line 8 (`argmax_{w⁻≥τ} w⁺`) | merge-loop greedy pick | Exact. |

Subtle deviations under `--approx` (today):

- `_count_intersection` iterates `b`, breaks on first match in `b'` — under
  approx matching this is not strictly symmetric.
- `negative_score` uses two different conflict-set sizes as numerators.

Both vanish under `--no_approx`. **No code change to make scoring strict — we
default to `--no_approx` for the paper-strict path and document approx
behavior.**

Active deviations in the previous design draft (now removed):

- A `MAX_BUCKET_SIZE` stopword cap was proposed as a memory mitigation. **It
  is not in the paper. Dropped.** The Counter for `theta_overlap` may OOM at
  Vertica 30k scale; we accept that as the honest single-machine limit.

## Phase A1 — `--theta_edge` filter

**Change site:** `_compute_initial_scores` in `synthesis.py`, and the
parallel mirror in `parallel_pipeline.py`.

Today:

```python
if ps > 0:
    pos_scores[key] = ps
    positive_edges += 1
```

After:

```python
if ps >= theta_edge:
    pos_scores[key] = ps
    positive_edges += 1
```

`theta_edge` is added as a parameter (default `0.85`) to:

- `_compute_initial_scores`
- `parallel_compute_initial_scores`
- `greedy_partition`
- `parallel_greedy_partition`

CLI: `--theta_edge` flag in `main.py`, default `0.85`.

**Boundary semantics.** `>=` not `>`, matching the paper's inclusive
description of best performance at 0.85 and ensuring identical pair sets
(`w⁺ = 1.0`) are always kept.

## Phase A2 — `--theta_overlap` properly applied

**The current bug.** `theta_overlap` is plumbed through `greedy_partition`
but **never checked**. Every candidate pair sharing ≥1 entry in `pair_index`
or `left_index` is admitted today.

**Implementation.** Replace `_build_overlap_set(pair_index, left_index)`
with `_build_overlap_set(pair_index, left_index, theta_overlap)` that
counts per-edge co-occurrence in each index and admits only edges where
`count > theta_overlap`:

```python
from collections import Counter

def _build_overlap_set(pair_index, left_index, theta_overlap):
    pair_count: Counter[Tuple[int, int]] = Counter()
    for indices in pair_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = (indices[x], indices[y]) if indices[x] < indices[y] \
                       else (indices[y], indices[x])
                pair_count[(a, b)] += 1

    left_count: Counter[Tuple[int, int]] = Counter()
    for indices in left_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = (indices[x], indices[y]) if indices[x] < indices[y] \
                       else (indices[y], indices[x])
                left_count[(a, b)] += 1

    return {k for k, v in pair_count.items() if v > theta_overlap} | \
           {k for k, v in left_count.items() if v > theta_overlap}
```

**Default change:** `theta_overlap` default in `main.py` changes from `1` to
`3` (paper §5.4: "quality of resulting clusters are insensitive to
`theta_overlap`"; >1 is required for the filter to do anything; 3 is the
midpoint of their sensitivity sweep).

**Memory honesty.** The Counter scales with raw enumerations. At Vertica
10k (~333 M edges) the Counter is ~50 GB of dict overhead — fits on dama's
251 GB, breaks on a laptop. At Vertica 30k (multi-billion edges) it
exceeds dama's RAM. **No MAX_BUCKET_SIZE cap** — paper-faithful. Document
the limit in the spec; revisit only if 30k becomes a target.

## Phase A3 — `negative_score` single `|F|`

**Current code** computes `f_from_b` and `f_from_b_prime` separately and
uses them as different numerators. Paper Eq 4 uses a single `|F|`:

```
w⁻(B, B') = -max( |F| / |B| , |F| / |B'| )
where F = {l | (l,r) ∈ B, (l,r') ∈ B', r ≠ r'}
```

**Change.** Compute the conflict set once:

```python
def negative_score(b, b_prime, use_approx=True):
    if not b or not b_prime:
        return 0.0
    f = _conflict_set(b, b_prime, use_approx)
    return -max(len(f) / len(b), len(f) / len(b_prime))
```

**Fidelity claim.** Under `--no_approx`, the previous code's
`|f_from_b| == |f_from_b_prime|` (both index the same set of
conflicting `l` values via different iteration paths). So the change is a
no-op under exact matching. Under approx the change picks one canonical F
direction; the paper does not specify behavior under approx so this is the
narrowest deviation from current code that matches Eq 4 exactly.

A test (`test_negative_score_strict_mode_no_op_under_exact`) pins the
no-op property.

## Phase B — Connected-component decomposition

**Algorithm** (paper §4.2 + Appendix E; we use Union-Find with path
compression on a single machine, equivalent to Hash-to-Min on Map-Reduce):

```python
def _connected_components(pos_scores, n_candidates):
    """Components of the positive-edge graph.

    Returns a list of sets, each holding candidate indices in one
    component. Candidates with no positive edges form singletons.
    Components are returned in `min(indices)` order for determinism.
    """
    parent = list(range(n_candidates))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
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

Cost: O(|pos_scores| · α(n_candidates)) — practically linear.

**Restructured `_run_merge_loop`.** Today's monolithic merge loop is split:

```python
def _run_merge_loop(candidates, pos_scores, neg_scores, tau,
                    theta_overlap, output_folder):
    # 1. Save edge scores (unchanged; before component split).
    _save_edge_scores(pos_scores, neg_scores, output_folder)

    # 2. Connected components.
    components = _connected_components(pos_scores, len(candidates))
    print(f"  Found {len(components)} components (largest: "
          f"{max(len(c) for c in components)})")

    # 3. Bucket scores by component in one pass.
    cand_to_comp: Dict[int, int] = {c: i for i, comp in enumerate(components)
                                    for c in comp}
    comp_pos: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    comp_neg: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    for key, v in pos_scores.items():
        comp_pos[cand_to_comp[key[0]]][key] = v
    for key, v in neg_scores.items():
        a, b = key
        if cand_to_comp.get(a) == cand_to_comp.get(b):
            comp_neg[cand_to_comp[a]][key] = v
        # else: cross-component neg edges can never block a merge — dropped.

    # 4. Greedy merge per component.
    partitions: List[Partition] = []
    for i, component in enumerate(components):
        if len(component) == 1:
            partitions.append([next(iter(component))])
            continue
        partitions.extend(_greedy_merge_component(
            candidates, component, comp_pos[i], comp_neg[i], tau,
        ))
    return partitions
```

**`_greedy_merge_component`** is today's merge-loop body lifted into a
function operating only on the indices in `component`. `part_members`,
`part_pairs`, and the per-iteration `pos_scores.items()` scan are scoped to
this one component. Returns `List[Partition]`.

**Determinism.** Components are sorted by `min(indices)`. Within each
component, the existing `sorted(pos_scores.items())` ordering and tie-break
rules carry over unchanged. Cross-component output is deterministic.

**Negative edges across components.** A neg edge whose endpoints sit in
different positive-edge components can never block a merge (no positive
path between the partitions to merge in the first place). Dropping them at
the bucketing step is a real memory and per-iteration win and is correct.

**Why this is the dominant fix.** Today's `5 s/round × 73 k merges ≈ 4
days` projection is for one giant merge loop on 70 M positive edges. With
components, each iteration scans `|component pos_scores|` instead — for
typical fragmentation that's 3-4 orders of magnitude smaller, putting the
10k Vertica run in the "tens of minutes" range.

## Paper-strict mode

Document in the spec (and in the CLI help text for `--no_approx`):

> Run `main.py --no_approx --parallel_workers N --theta_edge 0.85
> --theta_overlap 3` for paper-strict behavior. Under exact matching the
> scoring functions are bit-for-bit Equation 3 and Equation 4 from
> Wang & He (SIGMOD 2017). Under `--approx`, `_count_intersection` and the
> dual-direction conflict counting in earlier versions are extensions not
> specified by the paper.

No code change required for the strict mode — the existing `--no_approx`
flag suffices once Phase A3 lands.

## Architecture

### Files touched

| File | Change |
|---|---|
| `synthesis.py` | `_build_overlap_set` takes `theta_overlap`. `_compute_initial_scores` takes `theta_edge`, filters at `ps >= theta_edge`. `negative_score` uses single `|F|`. New `_connected_components`. `_run_merge_loop` restructured into per-component dispatch + lifted `_greedy_merge_component` helper. `greedy_partition` signature gains `theta_edge`. |
| `parallel_pipeline.py` | `parallel_compute_initial_scores` and `parallel_greedy_partition` accept and thread `theta_edge`. (No per-component parallelism in this spec.) |
| `main.py` | New `--theta_edge` flag (default 0.85). `--theta_overlap` default 1 → 3. `--no_approx` help text updated to mention paper-strict mode. |
| `conftest.py` | Add `synthesis_components_fixture` — 6 candidates in two disjoint positive-overlap clusters plus one isolated singleton. |
| `tests/test_parallel_synthesis.py` | Expand: theta_edge tests, theta_overlap tests, single-|F| no-op test, component correctness tests, with-components ≡ without-components equivalence, updated `EXPECTED_PARTITIONS` for the new defaults. |

### Public-API impact

- `greedy_partition` and `parallel_greedy_partition` gain `theta_edge=0.85`
  parameter (keyword-only after `use_approx`).
- `_compute_initial_scores` and `parallel_compute_initial_scores` gain
  `theta_edge` (passed from parent).
- `_build_overlap_set` signature gains `theta_overlap` (replaces today's
  two-arg form).
- `negative_score` signature unchanged; semantics narrow (one F).

## Testing

### Layer 1 — Per-fix units

| Test | Pins |
|---|---|
| `test_theta_edge_zero_keeps_all_positive_edges` | `theta_edge=0` ≡ today's `ps > 0` filter |
| `test_theta_edge_default_drops_low_weight_edges` | `theta_edge=0.85` on fixture; surviving edge count pinned |
| `test_theta_edge_one_keeps_only_perfect_overlap` | `theta_edge=1.0` keeps only identical pair sets |
| `test_theta_overlap_one_acts_like_today` | `theta_overlap=1` matches current discovery (every shared bucket entry → admitted) |
| `test_theta_overlap_high_filters_buckets` | Synthetic 2/3/4-share candidates, `theta_overlap=3` keeps only the 4-share pair |
| `test_negative_score_single_F_matches_paper_eq4` | Hand-crafted F; `-max(|F|/|B|, |F|/|B'|)` exact |
| `test_negative_score_strict_mode_no_op_under_exact` | Single-F vs old dual-F equal under `use_approx=False` |

### Layer 2 — Component correctness

| Test | Pins |
|---|---|
| `test_components_singletons_passthrough` | No-positive-edge candidates emerge as 1-element partitions, merge loop not entered |
| `test_components_disjoint_clusters_independent` | Two clusters → 2 components → no output partition spans both |
| `test_components_with_one_big_blob_equals_unsplit` | Fully-connected fixture → 1 component → identical output to today's monolithic merge loop |
| `test_components_cross_component_neg_edges_dropped` | Verify `comp_neg` returned by the bucketing helper omits cross-component neg keys |

### Layer 3 — Equivalence and regression

| Test | Pins |
|---|---|
| `test_refactor_preserves_behavior` (updated) | Re-capture `EXPECTED_PARTITIONS` for new defaults (`theta_edge=0.85`, `theta_overlap=3`) |
| `test_with_components_equals_without_components` | `theta_edge=0` keeps all edges → per-component output equals monolithic output. Critical regression test for Phase B. |
| `test_parallel_equals_sequential` (updated) | Same fixture, same defaults, `n_workers ∈ {2, 4}` |
| `test_parallel_determinism` (unchanged signature) | Re-run with identical input → identical output |

### Layer 4 — Manual verification on real corpora

Run after all unit tests pass:

1. WDC `output/candidates.jsonl` with new defaults — wall-clock from ~1.7 hr
   (parallel-only) to "tens of minutes". Partition counts will differ from
   May 20 baseline; document the diff.
2. Vertica 10k corpus — confirm "days → minutes" estimate. Record final
   partition count and a sample of synthesized mappings.
3. Vertica 30k (stretch) — if Counter for `theta_overlap` fits in 251 GB,
   capture full pipeline result. If OOM, document the limit and stop.

## Expected runtime impact

The dominant change is per-iteration cost in the merge loop:

| Setting | 10k Vertica observed | After Phase A + B (projected) |
|---|---|---|
| Initial overlap-graph build | OK (memory-bound but feasible) | Same memory cost or slightly less |
| Initial scoring (parallel, no-approx) | 20.7 min | Slightly less (fewer surviving edges from θ_edge) |
| Merge loop / round | ~5 s (scan 70 M edges) | ~µs–ms (scan one component's edges) |
| Merge loop total | ~100 hr | ~minutes |

On WDC, the parallel-only run was 102 min (95% scoring, 5% merge). Expect
~25-30 min total, dominated by scoring.

## Out of scope / future work

- Per-component parallelism (across components, not within) — multiprocessing
  Pool over `_greedy_merge_component` calls.
- Union-Find for `part_members` inside the merge loop (different data
  structure than the components-side Union-Find).
- Heap-based `argmax` scan replacing `for key, ps in pos_scores.items()` —
  may become attractive at component sizes ≥ 100k entries.
- Bucket-cap memory mitigation for Counter (the dropped `MAX_BUCKET_SIZE`).
  Revisit only if Vertica 30k becomes a target and we can't get the paper's
  Map-Reduce approach.
