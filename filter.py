"""Threshold-based filtering, JSONL output, and reporting for WP1.

Inputs come from `npmi.score_corpus`; outputs are
- a partition of scored columns into kept/removed,
- a rebuilt corpus where each retained table contains only kept columns,
- a JSONL file mirroring the input schema with the rejected columns
  removed and added `coherence_scores` / `rejected_column_indices` fields,
- a coherence histogram (PNG) and a threshold sweep summary.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

from data_loader import Table
from npmi import ScoredColumn


def filter_corpus(
    scored: List[ScoredColumn],
    threshold: float = 0.3,
) -> Tuple[List[ScoredColumn], List[ScoredColumn]]:
    """Partition scored columns at `threshold`. Returns (kept, removed)."""
    kept: List[ScoredColumn] = []
    removed: List[ScoredColumn] = []
    for entry in scored:
        if entry[3] >= threshold:
            kept.append(entry)
        else:
            removed.append(entry)
    return kept, removed


FilteredTable = Tuple[Dict, List[List[str]], List[float], List[int]]


def rebuild_filtered_corpus(
    corpus: List[Table],
    kept: List[ScoredColumn],
) -> List[FilteredTable]:
    """Reconstruct each table with only its surviving columns.

    Tables with zero surviving columns are dropped. Returns a list of
    (metadata, kept_columns, kept_scores, rejected_column_indices) tuples
    in the original table order.
    """
    by_table: Dict[int, List[Tuple[int, List[str], float]]] = {}
    for ti, ci, values, score in kept:
        by_table.setdefault(ti, []).append((ci, values, score))

    out: List[FilteredTable] = []
    for ti, (metadata, columns) in enumerate(corpus):
        kept_for_table = sorted(by_table.get(ti, []), key=lambda x: x[0])
        if not kept_for_table:
            continue
        kept_indices = {c[0] for c in kept_for_table}
        kept_columns = [c[1] for c in kept_for_table]
        kept_scores = [c[2] for c in kept_for_table]
        rejected = [i for i in range(len(columns)) if i not in kept_indices]
        out.append((dict(metadata), kept_columns, kept_scores, rejected))
    return out


def save_filtered_corpus(filtered: List[FilteredTable], output_path: str) -> None:
    """Write the filtered corpus as JSON Lines.

    Each record mirrors the input schema with two added fields:
      - `coherence_scores`: list[float], one per surviving column
      - `rejected_column_indices`: list[int], original column indices removed
    The `relation` field is replaced with the surviving columns only.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metadata, columns, scores, rejected in filtered:
            rec: Dict = dict(metadata)
            rec["relation"] = columns
            rec["coherence_scores"] = scores
            rec["rejected_column_indices"] = rejected
            f.write(json.dumps(rec) + "\n")


def filtering_report(
    kept: List[ScoredColumn],
    removed: List[ScoredColumn],
) -> None:
    """Print the filtering report described in the spec."""
    total = len(kept) + len(removed)
    pct = (len(removed) / total * 100) if total else 0.0
    print(f"  Total columns before filtering: {total}")
    print(f"  Total columns after filtering: {len(kept)}")
    print(f"  Removed: {len(removed)} ({pct:.1f}%)")
    print("  Examples of removed columns:")
    for ti, ci, vals, score in sorted(removed, key=lambda x: x[3])[:5]:
        sample = vals[:5]
        more = "..." if len(vals) > 5 else ""
        print(f"    table {ti} col {ci} score={score:.3f} values={sample}{more}")
    print("  Examples of kept columns:")
    for ti, ci, vals, score in sorted(kept, key=lambda x: x[3], reverse=True)[:5]:
        sample = vals[:5]
        more = "..." if len(vals) > 5 else ""
        print(f"    table {ti} col {ci} score={score:.3f} values={sample}{more}")


def threshold_sweep(
    scored: List[ScoredColumn],
    thresholds: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    output_path: str | None = None,
) -> str:
    """Print and (optionally) save a kept/removed table over multiple thresholds.

    Returns the rendered table as a string for downstream callers.
    """
    total = len(scored)
    lines = ["Threshold | Kept | Removed | Kept %",
             "----------+------+---------+-------"]
    for t in thresholds:
        kept = sum(1 for _, _, _, s in scored if s >= t)
        removed = total - kept
        pct = (kept / total * 100) if total else 0.0
        lines.append(f"   {t:.1f}    | {kept:>4} |  {removed:>4}   | {pct:.1f}%")
    rendered = "\n".join(lines)
    print(rendered)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    return rendered


def plot_coherence_distribution(
    scored: List[ScoredColumn],
    threshold: float,
    output_path: str,
) -> None:
    """Save a histogram of coherence scores with a vertical threshold line.

    Uses a non-interactive matplotlib backend so the call works in headless
    environments (CI, no display).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = [s for _, _, _, s in scored]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=40, range=(-1.0, 1.0))
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"threshold={threshold}")
    ax.set_xlabel("Coherence (mean pairwise NPMI)")
    ax.set_ylabel("Column count")
    ax.set_title(
        f"Column coherence distribution (N={len(scored)}, threshold={threshold})"
    )
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
