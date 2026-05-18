"""Conflict Resolution — WP4 (paper §4.3).

Implements Algorithm 4 from Wang & He (SIGMOD 2017) Appendix G.
Takes synthesized mappings from WP3 (synthesized_mappings.jsonl) and
removes conflicting value pairs to produce clean, consistent mappings.

A conflict occurs when the same left-hand value maps to two different
right-hand values within the same partition, violating the definition
of a mapping relationship.

Algorithm:
1. Find all conflicting value pairs in the partition
2. Iteratively remove the value pair that conflicts with the most others
3. Repeat until no conflicts remain
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MappingTable = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def find_conflicts(pairs: MappingTable) -> Dict[str, List[str]]:
    """Find all left values that map to more than one right value.

    Returns a dict of {left_value: [right_value1, right_value2, ...]}
    for every left value that has conflicts.
    """
    left_to_rights: Dict[str, Set[str]] = defaultdict(set)
    for left, right in pairs:
        left_to_rights[left].add(right)

    return {
        left: sorted(rights)
        for left, rights in left_to_rights.items()
        if len(rights) > 1
    }


def has_conflicts(pairs: MappingTable) -> bool:
    """Return True if there are any conflicting pairs."""
    return len(find_conflicts(pairs)) > 0


# ---------------------------------------------------------------------------
# Conflict resolution (Algorithm 4)
# ---------------------------------------------------------------------------

def resolve_conflicts(pairs: MappingTable) -> MappingTable:
    """Remove conflicting value pairs using Algorithm 4 from the paper.

    Iteratively finds the value pair (left, right) that participates in
    the most conflicts and removes it, until no conflicts remain.

    Args:
        pairs: list of (left, right) value pairs, possibly with conflicts.

    Returns:
        A new list of (left, right) pairs with no conflicts.
    """
    # Work with a mutable copy
    current: List[Tuple[str, str]] = list(pairs)

    while True:
        conflicts = find_conflicts(current)
        if not conflicts:
            break

        # Count how many conflicts each (left, right) pair is involved in
        conflict_count: Dict[Tuple[str, str], int] = defaultdict(int)
        for left, rights in conflicts.items():
            for right in rights:
                if (left, right) in [p for p in current]:
                    conflict_count[(left, right)] += len(rights) - 1

        if not conflict_count:
            break

        # Remove the pair involved in the most conflicts
        worst_pair = max(conflict_count, key=lambda p: conflict_count[p])
        current = [p for p in current if p != worst_pair]

    return current


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_synthesized_mappings(path: str) -> List[dict]:
    """Load synthesized_mappings.jsonl produced by WP3."""
    mappings = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["pairs"] = [tuple(p) for p in rec["pairs"]]
            mappings.append(rec)
    return mappings


def save_resolved_mappings(mappings: List[dict], output_path: str) -> None:
    """Save resolved mappings as JSONL. One line per mapping.

    Each line has the schema:
    {
        "partition_id": int,
        "pairs": [[left, right], ...],
        "size": int,
        "num_conflicts_removed": int
    }
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in mappings:
            fh.write(json.dumps(rec) + "\n")


def resolution_report(mappings: List[dict]) -> None:
    """Print a human-readable summary of conflict resolution results."""
    total = len(mappings)
    had_conflicts = sum(1 for m in mappings if m["num_conflicts_removed"] > 0)
    total_removed = sum(m["num_conflicts_removed"] for m in mappings)

    print(f"  Conflict resolution report:")
    print(f"  Total mappings processed: {total}")
    print(f"  Mappings with conflicts:  {had_conflicts}")
    print(f"  Total pairs removed:      {total_removed}")

    if had_conflicts > 0:
        print(f"  Examples of resolved mappings:")
        shown = 0
        for m in mappings:
            if m["num_conflicts_removed"] > 0:
                sample = m["pairs"][:3]
                print(
                    f"    partition {m['partition_id']}: "
                    f"removed {m['num_conflicts_removed']} pair(s), "
                    f"kept {m['size']} pairs. "
                    f"sample: {', '.join(f'({l},{r})' for l,r in sample)}"
                )
                shown += 1
                if shown >= 3:
                    break


# ---------------------------------------------------------------------------
# Main entry point for this module
# ---------------------------------------------------------------------------

def run_conflict_resolution(
    synthesized_path: str,
    output_path: str,
) -> List[dict]:
    """Load WP3 output, resolve conflicts, save and return results.

    Args:
        synthesized_path: path to synthesized_mappings.jsonl from WP3.
        output_path: where to write resolved_mappings.jsonl.

    Returns:
        List of resolved mapping dicts.
    """
    mappings = load_synthesized_mappings(synthesized_path)
    print(f"  Loaded {len(mappings)} synthesized mappings")

    resolved = []
    for rec in mappings:
        original_pairs = rec["pairs"]
        clean_pairs = resolve_conflicts(original_pairs)
        resolved.append({
            "partition_id": rec["partition_id"],
            "pairs": [list(p) for p in clean_pairs],
            "size": len(clean_pairs),
            "num_conflicts_removed": len(original_pairs) - len(clean_pairs),
        })

    resolution_report(resolved)
    save_resolved_mappings(resolved, output_path)
    print(f"  Saved resolved mappings to {output_path}")

    return resolved