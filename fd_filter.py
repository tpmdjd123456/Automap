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
    from tqdm import tqdm
    candidates: List[Candidate] = []
    excluded_metadata_keys = {"relation", "coherence_scores", "rejected_column_indices"}
    records_list = list(filtered_records)
    for table_idx, record in enumerate(tqdm(
        records_list, desc="FD filtering", unit="table", mininterval=30.0,
    )):
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
                        "source_metadata": dict(source_metadata),
                    })
    return candidates


def save_candidates(candidates: List[Candidate], output_path: str) -> None:
    """Write JSONL, one candidate per line. Creates parent directory
    if missing."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")


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
