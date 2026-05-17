# Deviations from `wp1_prompt.md`

This document records every place WP1's design departs from the literal text of `claude/wp1_prompt.md`, and why. The paper (Wang & He, *Synthesizing Mapping Relationships Using Table Corpus*, SIGMOD 2017, Section 3.1) is the ground truth; the prompt is a useful scaffold. Where they conflict, the paper wins.

## Summary table

| # | `wp1_prompt.md` says | We do | Why |
|---|---|---|---|
| 1 | Input is CSV files in a folder | Pluggable loader: JSONL (primary, matches `data/sample.json`) + CSV folder (secondary). Auto-dispatch on path. | Actual sample data is JSONL web-table corpus, not CSVs. CSV path retained for arbitrary local data. |
| 2 | "Skip columns with more than 100 unique values" | No size cap | Not in paper. Paper's whole premise is that PMI itself filters incoherent columns; an arbitrary size cap risks dropping legitimate large mapping columns (countries ~250, ISO codes ~250, airports >1000). |
| 3 | "For columns with more than 50 unique values, sample 50 random pairs" | All pairs (full enumeration) | Not in paper. Paper computes exact mean. Sample corpus is small enough that O(n²) is fine. Configurable later if scaling demands. |
| 4 | No filtering by table type | Only `tableType=="RELATION"` tables loaded; LAYOUT/MATRIX/etc. dropped | Paper assumes a relational table corpus. LAYOUT tables are HTML formatting noise (sidebars, navigation), not data. Including them pollutes the PMI signal. |
| 5 | No header handling | Strip header row when `hasHeader=true` (using `headerRowIndex`) | Not in paper but obvious oversight. Without stripping, column names like "country", "code" become "values" and inflate PMI artificially. |
| 6 | `load_table(filepath)` listed as a top-level function | Internal helper inside `_load_csv_folder`; primary entry point is `load_corpus(path)` | `load_corpus` auto-dispatches on path (file → JSONL, dir → CSV folder) for cleaner CLI. |
| 7 | Save filtered corpus as one CSV per table in `output_folder/` | One `filtered_corpus.jsonl` file (mirrors input schema with rejected columns removed) | Preserves web-table metadata (URL, page title, table type), avoids millions-of-tiny-files filesystem pressure at scale, hands off cleanly to WP2. |
| 8 | Index in JSON (pickle as fallback if too slow) | Pickle as primary | JSON cannot natively represent tuple keys for `(u,v)` pairs — would force lossy string-encoding. Pickle is also ~10× faster to load on large indexes. |
| 9 | Generate 20 synthetic CSV fixtures for testing | In-code synthetic mini-corpus inside `tests/test_wp1.py`; real validation runs on `data/sample.json` | Real corpus is already present. The 20-CSV fixture is redundant. In-code synthetic data keeps tests self-contained. |
| 10 | CLI flag `--corpus_folder` | CLI flag `--corpus_path` | Path can be a file (JSONL) or a folder (CSVs); the more general name fits both. |
| 11 | `threshold_sweep` returns `None` per spec | Returns `str` (rendered table) and optionally writes to disk | Lets `main.py` use the same function for both stdout printing and file writing without duplication |
| 12 | `test_npmi` function name | Kept (per prompt) but conflicts with pytest's `test_*` collection convention | The prompt explicitly named it `test_npmi`. Mitigation: pytest's default `python_files = test_*.py` excludes the top-level `npmi.py`, and call sites alias on import (`from npmi import test_npmi as npmi_sanity`) |
| 13 | Synthetic test corpus | Expanded from 6 tables to 16 in conftest.py | Original 6-table fixture was too small for NPMI to differentiate coherent from incoherent columns (every garbage pair scored +1 because each garbage value appeared in only one column). Added 10 noise tables (dates/colors/movies/percentages/greetings) so each garbage value has unrelated coherent contexts |

## What is NOT a deviation (preserved verbatim from prompt)

- The math: PMI, NPMI, coherence-as-mean-pairwise-NPMI, filter-by-threshold.
- The 5-module split: `data_loader.py`, `cooccurrence_index.py`, `npmi.py`, `filter.py`, `main.py`.
- Default threshold `0.3`.
- Threshold sweep over `{0.1, 0.2, 0.3, 0.4, 0.5}`.
- Coherence histogram with vertical threshold line.
- Stage banners `[Stage k/4]`, per-stage timing, progress prints.
- The filtering report (totals before/after, percentage removed, 5 examples each side).
- Docstrings + type hints on every public function, module-level docstrings.
- Allowed-libraries list.
- NPMI clipping to `[-1, +1]`.
- NPMI = -1 when `p(u,v) = 0`.
- Skip empty columns and columns with <2 unique values in the loader.
- `defaultdict(int)` for the index.
- Pair keys stored sorted so `(u,v)` and `(v,u)` collide.

---

## WP2 — Section 3.2 (Column-Pair Filtering by FD)

The original prompt (`wp1_prompt.md`) only covered Section 3.1. WP2 was
designed against the paper directly. Notable choices:

| # | Choice | Rationale |
|---|---|---|
| 14 | **JSONL output with deduplicated `(l, r)` pairs** | Matches paper §4.1 data model `B = {(l_i, r_i)}` directly. WP3 will need this set form for compatibility scoring. |
| 15 | **WP1 row-alignment patch** (`""` markers preserved in columns; PMI/coherence skip them) | The original loader dropped empties per column independently, breaking row alignment. PMI didn't notice (intra-column only) but FD requires aligned columns. WP1 filter decisions are unchanged (zero drift on real corpus). |
| 16 | **`min_rows = 3` for FD eligibility** | Below 3 non-empty rows, FD score is meaningless. Configurable via `--min_rows`. |
| 17 | **Reject pairs with <2 distinct values on either side** | Constant columns (Y always the same) trivially "satisfy" FD but carry no mapping signal. Same spirit as WP1's "<2 unique" column rule. |
| 18 | **Greedy witness-subset construction** | For each distinct x, pick the most-common y. Provably the largest R̄ — picking any other y for a given x covers fewer rows. |
| 19 | **No reverse-direction inference** | (C_i, C_j) and (C_j, C_i) evaluated independently. A 1:1 mapping survives both directions; an N:1 mapping survives one. Matches the paper's distinction between 1:1 and N:1 (§2.1). |

## Open items deferred to WP3+

- Section 4 (Table Synthesis): stitching candidates that share the same relationship.
- Section 5 (Conflict Resolution): post-processing to remove contradictions in synthesized mappings.
- Streaming / out-of-core processing for >sample-scale corpora.
- Distributed execution (paper uses MapReduce).
- Bidirectional FD inference (treating `X →_θ Y` and `Y →_θ X` as a single 1:1 mapping). Each direction stays independent in the WP2 output; WP3 can later detect 1:1 by finding both directions present.