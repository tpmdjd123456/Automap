"""Majority Voting — alternative conflict resolution (paper §5.6).

Instead of iteratively removing the most conflicting pair (Algorithm 4),
majority voting keeps the right value that appears most frequently
across source tables for each conflicting left value.

This is used as a comparison baseline against Algorithm 4 in the paper.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

MappingTable = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Core majority voting logic
# ---------------------------------------------------------------------------

def count_pair_support(
    candidate_indices: List[int],
    candidates: List[dict]
) -> Dict[Tuple[str, str], int]:
    """Count how many source tables support each (left, right) pair.
    
    Args:
        candidate_indices: indices of candidates in this partition
        candidates: all candidates loaded from candidates.jsonl
        
    Returns:
        dict mapping (left, right) -> count of supporting tables
    """
    support: Dict[Tuple[str, str], int] = defaultdict(int)
    for idx in candidate_indices:
        if idx < len(candidates):
            for pair in candidates[idx]["pairs"]:
                support[tuple(pair)] += 1
    return support


def majority_vote(
    pairs: MappingTable,
    support: Dict[Tuple[str, str], int]
) -> MappingTable:
    """Resolve conflicts using majority voting.
    
    For each left value that maps to multiple right values,
    keep the right value supported by the most source tables.
    
    Args:
        pairs: list of (left, right) pairs possibly with conflicts
        support: dict mapping (left, right) -> count of supporting tables
        
    Returns:
        clean list of (left, right) pairs with no conflicts
    """
    # Group right values by left value
    left_to_rights: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for left, right in pairs:
        # Use support count, default to 1 if not found
        count = support.get((left, right), 1)
        left_to_rights[left][right] = count

    # For each left value, keep the right value with highest support
    result = []
    for left, rights_counts in left_to_rights.items():
        best_right = max(rights_counts, key=lambda r: rights_counts[r])
        result.append((left, best_right))

    return sorted(result)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if "pairs" in rec:
                    rec["pairs"] = [tuple(p) for p in rec["pairs"]]
                results.append(rec)
    return results


def save_majority_voted(mappings: List[dict], output_path: str) -> None:
    """Save majority voted mappings as JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in mappings:
            out = dict(rec)
            out["pairs"] = [list(p) for p in out["pairs"]]
            f.write(json.dumps(out) + "\n")


def majority_voting_report(mappings: List[dict]) -> None:
    """Print a summary of majority voting results."""
    total = len(mappings)
    had_conflicts = sum(1 for m in mappings if m["num_conflicts_removed"] > 0)
    total_removed = sum(m["num_conflicts_removed"] for m in mappings)
    print(f"  Majority voting report:")
    print(f"  Total mappings processed : {total}")
    print(f"  Mappings with conflicts  : {had_conflicts}")
    print(f"  Total pairs removed      : {total_removed}")
    if had_conflicts > 0:
        print(f"  Examples:")
        shown = 0
        for m in mappings:
            if m["num_conflicts_removed"] > 0:
                print(
                    f"    partition {m['partition_id']}: "
                    f"removed {m['num_conflicts_removed']} pair(s), "
                    f"kept {m['size']} pairs. "
                    f"sample: {list(m['pairs'])[:2]}"
                )
                shown += 1
                if shown >= 3:
                    break


# ---------------------------------------------------------------------------
# Comparison with Algorithm 4
# ---------------------------------------------------------------------------

def compare_with_algorithm4(
    majority_results: List[dict],
    algorithm4_results: List[dict]
) -> None:
    """Compare majority voting results with Algorithm 4 results."""
    maj_removed = sum(m["num_conflicts_removed"] for m in majority_results)
    alg4_removed = sum(m["num_conflicts_removed"] for m in algorithm4_results)
    maj_kept = sum(m["size"] for m in majority_results)
    alg4_kept = sum(m["size"] for m in algorithm4_results)
    maj_conflicts = sum(1 for m in majority_results if m["num_conflicts_removed"] > 0)
    alg4_conflicts = sum(1 for m in algorithm4_results if m["num_conflicts_removed"] > 0)

    print(f"\n  {'Metric':<35} {'Majority Voting':>20} {'Algorithm 4':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Mappings with conflicts':<35} {maj_conflicts:>20} {alg4_conflicts:>15}")
    print(f"  {'Total pairs removed':<35} {maj_removed:>20} {alg4_removed:>15}")
    print(f"  {'Total pairs kept':<35} {maj_kept:>20} {alg4_kept:>15}")

    # Per-mapping comparison
    disagreements = 0
    alg4_lookup = {m["partition_id"]: m for m in algorithm4_results}
    for m in majority_results:
        alg4 = alg4_lookup.get(m["partition_id"])
        if alg4 and set(map(tuple, m["pairs"])) != set(map(tuple, alg4["pairs"])):
            disagreements += 1
    print(f"  {'Mappings where results differ':<35} {disagreements:>20}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_majority_voting(
    synthesized_path: str,
    candidates_path: str,
    output_path: str,
    algorithm4_path: str = None,
) -> List[dict]:
    """Run majority voting conflict resolution.
    
    Args:
        synthesized_path: path to synthesized_mappings.jsonl from WP3
        candidates_path: path to candidates.jsonl from WP2
        output_path: where to write majority_voted_mappings.jsonl
        algorithm4_path: optional path to resolved_mappings.jsonl for comparison
        
    Returns:
        List of resolved mapping dicts
    """
    synthesized = load_jsonl(synthesized_path)
    candidates = load_jsonl(candidates_path)
    print(f"  Loaded {len(synthesized)} synthesized mappings")
    print(f"  Loaded {len(candidates)} candidates")

    resolved = []
    for rec in synthesized:
        original_pairs = rec["pairs"]
        candidate_indices = rec.get("candidate_indices", [])

        # Count support for each pair from source tables
        support = count_pair_support(candidate_indices, candidates)

        # Apply majority voting
        clean_pairs = majority_vote(original_pairs, support)

        resolved.append({
            "partition_id": rec["partition_id"],
            "pairs": [list(p) for p in clean_pairs],
            "size": len(clean_pairs),
            "num_conflicts_removed": len(original_pairs) - len(clean_pairs),
        })

    majority_voting_report(resolved)

    # Compare with Algorithm 4 if provided
    if algorithm4_path and os.path.exists(algorithm4_path):
        print(f"\n  Comparison with Algorithm 4:")
        alg4 = load_jsonl(algorithm4_path)
        alg4_fixed = []
        for m in alg4:
            alg4_fixed.append({
                "partition_id": m["partition_id"],
                "pairs": [tuple(p) for p in m["pairs"]],
                "size": m["size"],
                "num_conflicts_removed": m["num_conflicts_removed"],
            })
        compare_with_algorithm4(resolved, alg4_fixed)

    save_majority_voted(resolved, output_path)
    print(f"\n  Saved majority voted mappings to {output_path}")
    return resolved