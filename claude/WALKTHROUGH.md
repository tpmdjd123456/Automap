# WP1 Code Walkthrough

A guided tour of the PMI coherence-filtering implementation. Read this top to bottom and you should be able to explain what each piece does and why it exists.

---

## 1. The problem we're solving

Auto-Map is a system that automatically synthesizes **mapping tables** like (Country → ISO Code) or (Company → Stock Ticker) by mining a large corpus of web tables. The full paper has three stages:

1. **Candidate extraction** (this is what we're implementing in WP1+WP2)
2. **Table synthesis** — stitching together fragments of the same mapping found in different tables (WP3)
3. **Conflict resolution** — handling cases where different tables disagree (WP4)

Within candidate extraction, the paper has two sub-steps. WP1 is the first one:

- **Step 3.1: Column filtering by PMI** — WP1 (done)
- **Step 3.2: Column-pair filtering by FD** — WP2 (done) ← we now cover both

The job of Step 3.1 is to throw away **incoherent columns** before we start enumerating column pairs. An incoherent column is one whose values don't semantically belong together — e.g. a column mixing dates, percentages, and proper nouns. Filtering these out early dramatically reduces the work for WP2 and prevents spurious mapping candidates downstream.

The cleverness: we measure coherence purely from co-occurrence statistics, with no domain knowledge or labeled data. If two values frequently appear together in the same column across many tables in the corpus, they're "the same kind of thing." If they don't, they aren't.

---

## 2. The math

We use **Pointwise Mutual Information** (PMI), normalized to a fixed range.

For a corpus of `N` columns and two values `u`, `v`:

```
p(u)    = (number of columns containing u) / N
p(u,v)  = (number of columns containing both u and v) / N

PMI(u,v) = log( p(u,v) / (p(u) * p(v)) )
```

PMI compares the actual joint probability `p(u,v)` to what it would be if u and v were independent (`p(u) * p(v)`). Positive PMI means "appear together more than chance"; negative means "less than chance."

PMI is unbounded, so the paper uses **Normalized PMI** (NPMI):

```
NPMI(u,v) = PMI(u,v) / -log(p(u,v))
```

This squashes everything into `[-1, +1]`:

- **+1**: u and v perfectly co-occur. Whenever you see one, you see the other.
- **0**: u and v are independent. Their joint frequency is exactly what chance predicts.
- **−1**: u and v never co-occur, even though they each appear in the corpus.

The **coherence of a column** is the mean NPMI over all unordered pairs of its distinct values:

```
S(C) = average of NPMI(v_i, v_j) over all pairs (v_i, v_j) in C, i < j
     = total / C(|distinct values|, 2)
```

Filter rule: keep columns where `S(C) >= τ`, drop the rest. Default `τ = 0.3`.

---

## 3. Edge cases the paper doesn't spell out

NPMI's formula breaks at boundaries that show up in real data:

| Situation | What we return | Why |
|---|---|---|
| u and v never co-occur | NPMI = -1 | The lower bound; "never seen together" is maximally incoherent. |
| Either u or v is unknown to the corpus | NPMI = -1 | Defensive — shouldn't happen in normal flow but can be called externally. |
| u and v in every single column (`p(u,v) = 1`) | NPMI = +1 | `−log(1) = 0` would divide-by-zero; we short-circuit. |
| Floating-point drift outside `[-1, +1]` | clipped | Final `max(-1, min(1, x))` absorbs `+1.0000000003` from numeric noise. |

These are all in `compute_npmi` in `npmi.py`. The pattern is: short-circuit the degenerate cases up front (saving a `log`), then run the math, then clip.

---

## 4. The pipeline

Four stages, one in-memory pass each (the index is the only thing that persists between runs):

```
JSONL/CSV
   │
   ▼
[Stage 1] data_loader  →  list[(metadata, columns)]
   │
   ▼
[Stage 2] cooccurrence_index  →  (cooc, value_count, N)  [pickled for re-use]
   │
   ▼
[Stage 3] npmi  →  list[(table_idx, col_idx, values, score)]
   │
   ▼
[Stage 4] filter  →  filtered_corpus.jsonl + histogram + sweep
```

The corpus and the index live in memory at the same time. That's fine for sample-scale data (~30 MB JSONL); a 100M-table production corpus would need streaming + MapReduce, which the paper handles but we don't.

---

## 5. The five modules

### 5.1 `data_loader.py` — corpus normalization

The single entry point is `load_corpus(path, *, table_types, strip_headers)`. It auto-dispatches on the path:

- **JSONL backend** (`_load_jsonl`): one record per line, with column-major `relation` and metadata fields. We filter to `tableType == "RELATION"` (configurable) because LAYOUT/MATRIX/etc tables are HTML formatting tricks, not data. We strip the header row when `hasHeader` is true so column names like "country" don't pollute the value index.
- **CSV folder backend** (`_load_csv_folder`): one CSV per table, row-major → transposed to columns. CSVs carry no metadata, so the first row is treated as data.

Every value passes through `clean_value` (strip + lowercase + collapse internal whitespace). Empty columns and columns with fewer than 2 unique values are dropped — they can't carry pairwise signal.

**Output shape:** `list[(metadata: dict, columns: list[list[str]])]`. The metadata dict is everything but `relation` from the original JSON record (or `{"source": filename}` for CSVs). Carrying metadata through is what lets the filtered output preserve `pageTitle`/`url`/`tableNum` for downstream provenance.

### 5.2 `cooccurrence_index.py` — global statistics

`build_cooccurrence_index(corpus)` makes one pass and returns a 3-tuple:

```python
Index = (
    cooccurrence: Dict[(str, str), int],   # how many columns contain BOTH values
    value_count:  Dict[str, int],          # how many columns contain each value
    total_columns: int,                    # N
)
```

Two things to know:

1. **We dedupe per column.** If "USA" appears 50 times in one column, `value_count["USA"]` increments by 1, not 50. This is what makes `p(u) = |C(u)| / N` correct as defined.
2. **Pair keys are sorted.** We store `(min(u,v), max(u,v))` so `(germany, france)` and `(france, germany)` collide on the same key.

Persistence is via pickle (not JSON) because tuple keys aren't natively JSON-serializable, and pickle is 10× faster on large indexes anyway. Re-runs reuse the cached pickle unless `--rebuild_index` is passed.

### 5.3 `npmi.py` — the math module

Four user-facing functions:

- `compute_pmi(u, v, index)` — raw PMI, returns `-inf` when any probability is zero. Mostly internal.
- `compute_npmi(u, v, index)` — normalized to `[-1, +1]`, with all the edge cases handled.
- `compute_coherence(column, index)` — mean NPMI over all unordered pairs of distinct values in the column, via `itertools.combinations(sorted(set(column)), 2)`.
- `score_corpus(corpus, index)` — applies `compute_coherence` to every column, returns `list[(table_idx, col_idx, values, score)]`.

There's also `test_npmi(index)`, a runtime sanity helper that prints the top co-occurring pair, checks symmetry on 100 pairs, and confirms range. It's called by `main.py` after the index is built; the name comes from the original prompt and unfortunately collides with pytest's `test_*` convention (we work around this by aliasing on import — see `claude/deviations.md` row 12).

The `_pair_key(u, v)` helper sorts (u, v) lexicographically before looking up in the cooccurrence dict — same canonicalization the index uses when building keys, so they match.

### 5.4 `filter.py` — partition + emit

This module turns scored columns into actual outputs. Six functions in pipeline order:

1. `filter_corpus(scored, threshold)` — partitions into `(kept, removed)`. Linear pass, simple comparison.
2. `rebuild_filtered_corpus(corpus, kept)` — for each table, keeps only the surviving columns and tracks which original indices were rejected. Tables with zero surviving columns are dropped entirely.
3. `save_filtered_corpus(filtered, output_path)` — writes JSONL. Each line mirrors the input schema with `relation` replaced (kept columns only) and adds `coherence_scores` (one per kept column) plus `rejected_column_indices` (which original positions were dropped).
4. `filtering_report(kept, removed)` — prints the before/after counts and 5 example columns from each side. Goes to stdout.
5. `threshold_sweep(scored, thresholds)` — renders the kept-vs-removed table for {0.1, 0.2, 0.3, 0.4, 0.5}, prints to stdout, optionally writes to disk. Lets you pick a threshold without re-scoring.
6. `plot_coherence_distribution(scored, threshold, output_path)` — histogram of all scores with a red dashed line at the threshold. Uses matplotlib's `Agg` backend so it works headless (in CI, in subprocess tests, on servers without displays).

The output JSONL is the hand-off to WP2: each line is a candidate-mapping-friendly table where every column has been confirmed coherent.

### 5.5 `main.py` — orchestration

argparse + four `[Stage k/4]` banners + per-stage timing. Loads cached index when possible; rebuilds when forced or absent. Calls `corpus_summary`, `index_summary`, `test_npmi` (runtime sanity), and `filtering_report` to print human-readable progress.

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
main.py (Section 3 driver)
   │
   ├── Stage 1: load corpus
   ├── Stage 2: build co-occurrence index
   ├── Stage 3: score columns (NPMI)
   ├── Stage 4: WP1 — filter columns by coherence
   │       → emits output/filtered_corpus.jsonl  (row-aligned columns)
   │
   └── Stage 5: WP2 — for each filtered table, for each ordered (i, j), i ≠ j:
           drop empty rows pairwise
           compute approximate FD via witness-subset construction
           if θ ≥ 0.95: emit candidate
       → emits output/candidates.jsonl  (one ordered column pair per line)
```

The new module is `fd_filter.py` (math + IO). `main.py` orchestrates all
five stages. The only change to WP1 itself was making the loader keep
`""` markers in columns to preserve row alignment (which PMI didn't
require but FD does). PMI/coherence functions skip `""` when iterating
distinct values, so the WP1 outputs are unchanged.

### Output schema

Each candidate is a JSON object on its own line, with eight fields:

| Field | Meaning |
|---|---|
| `pairs` | Deduplicated `(left, right)` value pairs from R̄. One entry per distinct left. |
| `theta` | The approximate-FD score, in `[0.95, 1.0]`. |
| `row_count` | `\|R\|` — non-empty rows used. |
| `covered_rows` | `\|R̄\|` — rows in the witness subset. |
| `source_table_index` | Line number in `filtered_corpus.jsonl`. |
| `left_column_index` | Index of left column in the filtered table's `relation`. |
| `right_column_index` | Index of right column. |
| `source_metadata` | Pass-through metadata (page title, URL, etc.). |

WP3 will treat each candidate's `pairs` as a set `B = {(l_i, r_i)}`
exactly as defined in paper §4.1, and find candidates B, B' that are
compatible (large overlap → same relationship → merge).

---

## 7. The synthetic test corpus

`conftest.py` defines `synthetic_corpus`, a 16-table fixture used by every unit test. Its design is non-obvious, so worth understanding.

The corpus has three categories:

1. **Coherent country/ticker tables** (5 tables, indices 0-4) — country↔ISO and ticker↔company mappings. The kind of column we want to keep.
2. **One garbage column** (index 5) — `["2024-01-01", "hello world", "83.5%", "the matrix", "blue"]`. Five values from totally unrelated domains. The kind of column we want to drop.
3. **Ten "noise" tables** (indices 6-15) — date columns, color columns, movie columns, percentage columns, greeting columns. **Each garbage value appears in two of these.** So "blue" appears in two color tables, "the matrix" in two movie tables, "2024-01-01" in two date tables, etc.

**Why the noise tables matter:** Without them, every garbage value would appear in exactly one column (the garbage column). When two values both appear in only one column and only together, NPMI is +1 by the formula — they "perfectly co-occur." So a garbage column with no noise context would score +1, indistinguishable from a coherent country column.

The noise tables put each garbage value in **multiple unrelated coherent columns**, which raises its individual frequency (`vc[blue] = 3`) without raising its co-occurrence with garbage neighbors (`cooc[(blue, the matrix)] = 1`). Now `p(u) * p(v)` is meaningfully larger than `p(u, v)`, so PMI is small, and NPMI for garbage pairs hovers around 0.28 — well below the 0.3 threshold.

This is a faithful miniature of how the math behaves on a real corpus. In the WDC sample corpus, "2024-01-01" appears in thousands of columns across the web; that's the same effect, just at scale.

The original synthetic corpus had 6 tables (no noise) and the test for "garbage column scores low" actually didn't pass — caught by the implementer during Task 7. Documented in `claude/deviations.md` row 13.

---

## 8. Key design decisions

These are choices we made that aren't obvious from the code and are documented in `claude/deviations.md`:

- **Paper-faithful math.** No size cap on column unique-values, no sampled-pairs approximation. The original prompt suggested both; we didn't, because the paper computes the exact mean.
- **Pluggable loader.** JSONL primary, CSV folder secondary. Real data is JSONL.
- **Strip headers + filter to RELATION.** The paper assumes a relational corpus; web tables have lots of LAYOUT/formatting noise that needs to go before NPMI sees it.
- **JSONL output with metadata.** One file, mirrors input schema, preserves URL/page-title for provenance. The original prompt said one CSV per table; we didn't, because that doesn't scale and loses metadata.
- **Pickle for the index.** JSON can't represent tuple keys; pickle is faster anyway.
- **Set-dedup per column.** Each column contributes +1 to a value's count regardless of how many times the value appears in that column. This is the correct interpretation of `p(u) = |C(u)| / N` from the paper.

---

## 9. What's NOT in WP1+WP2

- **Table synthesis** (paper §4) — WP3.
- **Conflict resolution** (paper §5) — WP4.
- **Streaming / out-of-core** — corpus is loaded into memory in one pass.
  Fine for sample-scale, would need rework for the paper's 100M-table
  experiments.
- **Distributed execution** — paper uses MapReduce. We run in-process.
- **Bidirectional FD inference** — `(C_i, C_j)` and `(C_j, C_i)` are
  evaluated as independent candidates. WP3 can later detect 1:1 mappings
  by finding both directions present in the candidate set.

---

## 10. Tying it all together

The flow when you run `python main.py --corpus_path data/sample.json ...`:

1. `data_loader.load_corpus` reads the JSONL, drops LAYOUT tables, strips headers, normalizes values, drops empty/short columns. Returns 2716 tables with 11956 total columns.
2. `cooccurrence_index.build_cooccurrence_index` makes one pass: for each column, dedupe values, increment `value_count[v]` per distinct value, increment `cooccurrence[(min, max)]` for each unordered pair. Result: `N=11956`, ~67k unique values, ~4.9M unique pairs.
3. `npmi.score_corpus` calls `compute_coherence` on every column. Each call: take distinct values, enumerate pairs, look up `cooccurrence` and `value_count`, compute NPMI, average. Returns 11956 scores.
4. `filter.filter_corpus` partitions at 0.3. ~11.6k columns pass; ~350 fail. `rebuild_filtered_corpus` walks each original table, keeps only surviving columns, drops empty tables. `save_filtered_corpus` writes the JSONL. `plot_coherence_distribution` writes the histogram. `threshold_sweep` writes the comparison table.

Total elapsed time: ~9 seconds.

---

## 11. If someone asks you a question

A few likely ones:

> **"Why NPMI instead of plain PMI?"**
> PMI is unbounded, so absolute thresholds like "0.3" wouldn't generalize across corpora. NPMI maps the same signal into `[-1, +1]` so a single threshold has consistent semantics regardless of corpus size.

> **"Why the average over pairs, not the median or min?"**
> The paper specifies the mean. The mean is also less brittle than the min — a single weakly-correlated pair could dominate. The mean lets occasional weird pairs be averaged out by the rest.

> **"Why dedupe values per column?"**
> Because the probabilities are defined as fractions of columns, not fractions of cells. A value that appears 50 times in one column is still in just one column, so `|C(u)|` increments by 1.

> **"Why is the threshold 0.3 and not, say, 0.5?"**
> Empirically. The paper validates downstream — they tune τ by checking which value gives the best mapping-quality results from WP2/WP3. We use 0.3 as the default but the threshold-sweep output lets you re-run downstream stages at other thresholds without rebuilding the index.

> **"Why does main.py take 9 seconds instead of 9 minutes?"**
> The corpus is 30 MB and 12k columns; the inner loop is C-level dict lookups and `math.log`. It's not slow because it's not big. The paper's 100M-table experiments need MapReduce; ours doesn't.

> **"Why does the histogram have a single tall bar near +1?"**
> The WDC RELATION tables in `data/sample.json` are mostly already coherent (the noise was filtered upstream by the WDC pipeline). PMI alone isn't doing dramatic filtering on this sample — that's expected. The paper achieves ~78% removal but only after combining PMI (§3.1) with FD (§3.2). PMI alone removes the worst 2-3% on a clean corpus.

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

---

## 12. WP3: Table Synthesis (paper §4.2)

### The problem

After WP2, we have thousands of small candidate two-column tables. Each 
one is a fragment of a mapping relationship. For example:
Table 1: (South Korea, KOR), (France, FRA), (Germany, DEU)
Table 2: (Korea Republic, KOR), (France, FRA), (Japan, JPN)
Table 3: (South Korea, KOR), (France, FRA), (Algeria, DZA)
Table 1 and Table 2 are about the same relationship (country → ISO code) 
but use different synonyms. Table 3 is about a different standard (ISO 
vs IOC). WP3 figures out which tables belong together and merges them.

### The key insight

Two tables that describe the same relationship should:
- Share many common value pairs (**positive compatibility**)
- NOT have conflicting pairs where the same left value maps to different 
  right values (**negative incompatibility**)

### The math (paper §4.1, Equations 3 and 4)

**Positive compatibility** uses Maximum-of-Containment:
w+(B, B') = max( |B ∩ B'| / |B| , |B ∩ B'| / |B'| )
This is high when one table is mostly contained in the other. Regular 
Jaccard similarity would unfairly penalize small tables being merged 
into large ones.

**Negative incompatibility** uses conflict ratio:
w-(B, B') = -max( |F| / |B| , |F| / |B'| )
Where F is the set of left values that map to different right values 
across the two tables. A negative score below threshold τ blocks the merge.

### Approximate string matching (Algorithm 2)

Real tables have minor variations like "Korea Republic" vs "Korea, 
Republic of". The paper uses edit distance with a fractional threshold:
threshold = min( floor(len(v1) * 0.2), floor(len(v2) * 0.2), 10 )
Short strings like "USA" require exact match (threshold=0). Longer 
strings allow small typos. Implemented in `approx_match()` in 
`synthesis.py` using band dynamic programming (Ukkonen-style) for 
efficiency.

### The greedy algorithm (Algorithm 3, paper §4.2)

The optimization problem is NP-hard (reduction from graph multi-cut). 
The greedy heuristic:
Start: each candidate table is its own partition
Find the pair of partitions with highest w+ where w- >= tau
Merge them into one partition
Update scores for the new partition vs all others
Repeat until no more merges are possible
Score updates after a merge are additive for positive scores and use 
minimum for negative scores — directly from Algorithm 3 in the paper.

### Inverted index for efficiency

Computing all pairwise scores naively is O(N²). Instead we build two 
inverted indexes:

- `pair_index`: maps (left, right) → list of candidate indices containing that pair
- `left_index`: maps left value → list of candidate indices containing it

Only candidates sharing at least one value pair need their compatibility 
computed. In practice this makes the number of edges much smaller than N².

### Output

Writes `synthesized_mappings.jsonl` to the output folder. Each line has:

| Field | Meaning |
|---|---|
| `partition_id` | Unique ID for this synthesized mapping |
| `candidate_indices` | Which WP2 candidates were merged |
| `pairs` | Union of all (left, right) pairs from merged candidates |
| `size` | Total number of unique pairs |
| `num_source_tables` | How many candidate tables were merged |

---

## 12. WP4: Conflict Resolution (paper §4.3)

### The problem

After WP3 synthesis, some merged mapping tables contain **conflicts** — 
the same left-hand value maps to two different right-hand values within 
the same partition. For example:
(Algeria, ALG)   ← from one table
(Algeria, DZA)   ← from a different table
Both survived WP3 because the tables were compatible enough to merge, 
but they violate the definition of a mapping relationship (one left value 
must map to exactly one right value).

### Why conflicts happen

When many tables are merged into one partition, some will have quality 
issues or extraction errors. For example, a table about IOC codes and a 
table about ISO codes might get merged because they share many country 
names. The conflicting pairs reveal the mistake.

### The algorithm (Algorithm 4, Appendix G)

The paper proves this problem is NP-hard (reduction from Maximum 
Independent Set). The greedy approximation works as follows:
Find all conflicts: left values that map to more than one right value
For each (left, right) pair, count how many other pairs it conflicts with
Remove the pair involved in the most conflicts
Repeat until no conflicts remain

This is implemented in `conflict_resolution.py`:

- `find_conflicts(pairs)` — returns dict of {left: [right1, right2, ...]} 
  for every conflicting left value
- `has_conflicts(pairs)` — quick boolean check
- `resolve_conflicts(pairs)` — runs the greedy algorithm, returns clean pairs
- `run_conflict_resolution(synthesized_path, output_path)` — full pipeline

### Output

Writes `resolved_mappings.jsonl` to the output folder. Each line has:

| Field | Meaning |
|---|---|
| `partition_id` | Which synthesized mapping this came from |
| `pairs` | Clean (left, right) pairs after conflict removal |
| `size` | Number of pairs remaining |
| `num_conflicts_removed` | How many pairs were removed |

### Effect on quality

From the paper (§5.6): conflict resolution improves precision from 0.903 
to 0.965 on average, while recall only drops slightly from 0.885 to 0.878. 
The biggest improvements are on mappings like (state → capital) that tend 
to get confused with similar mappings like (state → largest-city).
