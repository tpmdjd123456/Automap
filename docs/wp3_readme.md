# WP3 — Table Synthesis (paper §4)

Re-implementation of **Section 4** of Wang & He, *Synthesizing Mapping
Relationships Using Table Corpus* (SIGMOD 2017).

WP3 takes the candidate column pairs produced by WP2 (`candidates.jsonl`)
and groups them into coherent, conflict-free **synthesized mapping tables**
using a greedy partitioning algorithm.

---

## Overview

WP2 outputs many small, overlapping candidate tables — each is just a pair
of columns from one source table. WP3 merges compatible candidates into
larger unified mapping relationships (e.g. all "Country → ISO code"
fragments merge into one table), while keeping incompatible ones separate
(e.g. "Country → ISO2" and "Country → IOC code" stay apart if they conflict
on shared countries).

---

## Algorithm

### Step 1 — Approximate String Matching (§4.1, Algorithm 2)

Values are compared using **approximate edit distance** rather than exact
equality, so minor spelling variants (`"usa"` / `"u.s.a."`) are treated as
the same value.

The threshold is computed per the paper:

```
threshold = min( floor(len(v1) * fed), floor(len(v2) * fed), ked )
```

Defaults: `fed = 0.2` (fractional edit-distance factor), `ked = 10`
(absolute cap). Band dynamic programming (Ukkonen-style) is used for
efficiency — O(threshold × min(len1, len2)) instead of O(len1 × len2).

### Step 2 — Pairwise Compatibility Scores (§4.1, Equations 3 & 4)

For every pair of candidate tables B and B':

**Positive score** — Maximum-of-Containment similarity (how much do they overlap?):

$$w^+(B, B') = \max\!\left(\frac{|B \cap B'|}{|B|},\ \frac{|B \cap B'|}{|B'|}\right)$$

Range: `[0, 1]`. A score of 1.0 means one table is fully contained in the
other.

**Negative score** — Conflict penalty (do they disagree on any left-hand values?):

$$w^-(B, B') = -\max\!\left(\frac{|F|}{|B|},\ \frac{|F|}{|B'|}\right)$$

where `F` is the conflict set — left values where B and B′ map to different
right values. Range: `[-1, 0]`. A score of 0.0 means no conflicts.

### Step 3 — Greedy Partitioning (§4.2, Algorithm 3)

Starting with each candidate in its own partition:

1. Build an inverted index over all `(left, right)` pairs for efficiency.
2. Find all candidate pairs with shared value overlap.
3. Compute `w+` and `w-` for each overlapping pair.
4. Repeatedly merge the partition pair with the **highest `w+`**, provided
   `w- >= tau` (the negative threshold, default `−0.2`).
5. Repeat until no more valid merges exist.

Each partition becomes one synthesized mapping table — the union of all
pairs from its member candidates, deduplicated and sorted.

---

## Module: `synthesis.py`

| Function | Description |
|---|---|
| `load_candidates(path)` | Load `candidates.jsonl` from WP2; converts `pairs` from lists to tuples. |
| `approx_match(v1, v2, fed, ked)` | Band-DP edit distance with paper threshold formula. |
| `positive_score(b, b_prime, use_approx)` | Maximum-of-Containment w+ score. |
| `negative_score(b, b_prime, use_approx)` | Conflict-set w- score. |
| `build_inverted_index(candidates)` | Returns `(pair_index, left_index)` for fast lookup. |
| `greedy_partition(candidates, tau, theta_overlap, use_approx)` | Algorithm 3 — returns `List[List[int]]` (partitions of candidate indices). |
| `synthesize_mapping(partition, candidates)` | Unions pairs in a partition into one deduplicated, sorted mapping table. |
| `save_synthesized_mappings(partitions, candidates, output_path)` | Writes `synthesized_mappings.jsonl`. |
| `synthesis_report(partitions, candidates)` | Prints stats: total mappings, singletons, top-5 largest. |

---

## CLI (via `main.py`)

WP3 runs as Stage 6 of the full pipeline:

```bash
python main.py --corpus_path data/sample.json \
               --output_folder output/ \
               --threshold 0.1 \
               --theta 0.95 \
               --tau -0.2
```

### WP3-specific flags

| Flag | Default | Description |
|---|---|---|
| `--tau` | `-0.2` | Negative-weight threshold. Merges where `w- < tau` are blocked. Lower (more negative) = more permissive; less negative = stricter conflict avoidance. |
| `--theta_overlap` | `1` | Minimum shared value pairs required before computing full scores (efficiency filter). |
| `--no_approx` | off | Disable approximate string matching; use exact equality instead. |

---

## Output

`output/synthesized_mappings.jsonl` — one record per synthesized mapping:

```json
{
  "partition_id": 0,
  "candidate_indices": [0, 2],
  "pairs": [["canada", "can"], ["france", "fra"], ["germany", "deu"], ["japan", "jpn"], ["united states", "usa"]],
  "size": 5,
  "num_source_tables": 2
}
```

| Field | Description |
|---|---|
| `partition_id` | Sequential index of this mapping. |
| `candidate_indices` | Which WP2 candidates were merged into this mapping. |
| `pairs` | Sorted, deduplicated `[left, right]` value pairs. |
| `size` | Number of pairs in this mapping. |
| `num_source_tables` | How many distinct source tables contributed candidates. |

---

## Example run on `data/sample.json`

```
[Stage 6/6] Table synthesis (tau=-0.2)...
  Loaded 6 candidates
  pair_index: 16 unique pairs
  left_index: 16 unique left values
  Non-zero positive edges: 2
  Blocking negative edges (w- < tau): 0
  Merge round 1: merged partition sizes [1, 1] -> 2 (w+=0.333, w-=0.000)
  Merge round 2: merged partition sizes [1, 1] -> 2 (w+=0.333, w-=0.000)
  Converged after 2 merges. 4 partitions.

  Total synthesized mappings: 4
  Singleton partitions: 2 (50.0%)
  Multi-table partitions: 2
  Pairs per mapping: min=3 mean=4.0 max=5
```

**Mappings produced:**

| Mapping | Pairs | Source tables | Sample |
|---|---|---|---|
| Country → ISO | 5 | 2 | canada→can, france→fra, germany→deu … |
| ISO → Country | 5 | 2 | can→canada, deu→germany, fra→france … |
| Ticker → Company | 3 | 1 | aapl→apple, googl→alphabet, msft→microsoft |
| Company → Ticker | 3 | 1 | apple→aapl, alphabet→googl, microsoft→msft |

Country↔ISO candidates merged correctly across source tables. Ticker
candidates stayed as singletons because they share no pairs with the
country group.

---

## Tests

```bash
python -m pytest tests/test_wp3.py -v
```

53 tests covering:

| Area | Tests |
|---|---|
| `approx_match` | Exact match, threshold=0 short-circuits, length difference early-exit, band DP correctness, FED/KED parameter effects |
| `positive_score` | Empty inputs, identical tables (score=1), disjoint tables (score=0), partial overlap, asymmetric sizes |
| `negative_score` | No conflict (score=0), full conflict (score=−1), partial conflict, symmetry |
| `greedy_partition` | Empty input, all singletons, full merge, tau blocking, IOC/ISO paper example |
| `build_inverted_index` | pair_index correctness, left_index correctness, multi-candidate overlap |
| `synthesize_mapping / save / report` | Output schema validation, deduplication, sort order, file I/O, print output |

The **IOC/ISO paper example** is the key integration fixture — B0/B1 are
IOC tables (should merge), B2/B3/B4 are ISO tables (should merge), but
IOC≠ISO groups stay separate because "algeria" maps to "alg" vs "dza",
triggering a conflict that exceeds `tau`.

---

## Position in the Pipeline

```
data/sample.json
      │
      ▼  Stage 1 — data_loader.py
  corpus (16 tables, 21 columns)
      │
      ▼  Stage 2/3 — cooccurrence_index.py + npmi.py
  scored columns
      │
      ▼  Stage 4 — filter.py                    (WP1)
  filtered_corpus.jsonl  (8 columns kept)
      │
      ▼  Stage 5 — fd_filter.py                 (WP2)
  candidates.jsonl  (6 candidates)
      │
      ▼  Stage 6 — synthesis.py                 (WP3)  ◄── this module
  synthesized_mappings.jsonl  (4 mappings)
      │
      ▼  Stage 7 — conflict_resolution.py       (WP4, future)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Greedy merge terminates when no valid pair remains | Faithful to Algorithm 3; avoids unnecessary iterations. |
| Inverted index scopes pair comparison | Without it, O(n²) score computation would be prohibitive on large corpora. |
| Approximate matching is opt-in (`use_approx=True` default) | Exact-equality mode (`--no_approx`) retained for testing and debugging. |
| Partitions stored as lists of candidate indices | Defers pair union until output time; allows re-scoring merged partitions. |
| `synthesize_mapping` deduplicates and sorts | Output is deterministic regardless of merge order. |

---

## References

Wang, S. & He, B. (2017). *Synthesizing Mapping Relationships Using Table
Corpus*. SIGMOD 2017. Section 4: Table Synthesis, Algorithm 3, Equations 3–4.
