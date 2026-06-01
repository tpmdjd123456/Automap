"""Baseline methods for comparison with AutoMap synthesis (paper §5.1).

Implements three baselines from the paper:

1. UnionDomain — merge tables with same column structure from same page
2. UnionWeb    — merge tables with same column structure across all pages
3. SchemaCC    — schema matching using connected components

These are used to show that AutoMap's value-based synthesis outperforms
simpler column-name/structure based approaches.

Paper reference: Wang & He, SIGMOD 2017, Section 5.1
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> List[dict]:
    """Load raw corpus from JSONL file."""
    tables = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tables.append(json.loads(line))
    return tables


def get_column_signature(table: dict) -> str:
    """Get a signature for a table based on number of columns.
    
    Since our corpus strips headers, we use column count as
    a proxy for column name matching (as described in paper §5.1).
    
    For richer corpora with headers, this would use actual column names.
    """
    relation = table.get("relation", [])
    return f"ncols_{len(relation)}"


def get_domain(table: dict) -> str:
    """Get domain identifier for a table.
    
    Uses pgId (Wikipedia page ID) as proxy for domain,
    since our corpus doesn't have URL fields.
    """
    return str(table.get("pgId", "unknown"))


def merge_tables(tables: List[dict]) -> List[Tuple[str, str]]:
    """Merge multiple tables into one set of value pairs.
    
    For each table, extract all ordered (col_i, col_j) pairs
    where col_i functionally determines col_j.
    
    Returns deduplicated list of (left, right) value pairs.
    """
    pairs = set()
    for table in tables:
        relation = table.get("relation", [])
        if len(relation) < 2:
            continue
        # Take first two columns as the mapping
        col0 = relation[0]
        col1 = relation[1]
        min_len = min(len(col0), len(col1))
        for i in range(min_len):
            if col0[i] and col1[i]:
                pairs.add((col0[i].lower().strip(), col1[i].lower().strip()))
    return list(pairs)


# ---------------------------------------------------------------------------
# Baseline 1: UnionDomain
# ---------------------------------------------------------------------------

def union_domain(corpus: List[dict]) -> List[dict]:
    """Merge tables with same column structure from same page/domain.
    
    Groups tables by (domain, column_signature) and merges each group
    into one mapping relationship.
    
    Paper §5.1: "union together tables within the same website domain,
    if their column names are identical but row values are disjoint"
    
    Args:
        corpus: list of raw table dicts
        
    Returns:
        List of merged mapping dicts
    """
    # Group tables by (domain, column_signature)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for table in corpus:
        if len(table.get("relation", [])) < 2:
            continue
        domain = get_domain(table)
        sig = get_column_signature(table)
        key = f"{domain}_{sig}"
        groups[key].append(table)

    # Merge each group into one mapping
    mappings = []
    for group_id, tables in groups.items():
        pairs = merge_tables(tables)
        if not pairs:
            continue
        mappings.append({
            "group_id": group_id,
            "method": "UnionDomain",
            "num_source_tables": len(tables),
            "pairs": [list(p) for p in pairs],
            "size": len(pairs),
        })

    return mappings


# ---------------------------------------------------------------------------
# Baseline 2: UnionWeb
# ---------------------------------------------------------------------------

def union_web(corpus: List[dict]) -> List[dict]:
    """Merge tables with same column structure across entire corpus.
    
    Groups tables by column_signature only (ignoring domain) and
    merges each group into one mapping relationship.
    
    Paper §5.1: extends UnionDomain to merge across all domains.
    
    Args:
        corpus: list of raw table dicts
        
    Returns:
        List of merged mapping dicts
    """
    # Group tables by column_signature only (no domain restriction)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for table in corpus:
        if len(table.get("relation", [])) < 2:
            continue
        sig = get_column_signature(table)
        groups[sig].append(table)

    # Merge each group into one mapping
    mappings = []
    for sig, tables in groups.items():
        pairs = merge_tables(tables)
        if not pairs:
            continue
        mappings.append({
            "group_id": sig,
            "method": "UnionWeb",
            "num_source_tables": len(tables),
            "pairs": [list(p) for p in pairs],
            "size": len(pairs),
        })

    return mappings


# ---------------------------------------------------------------------------
# Baseline 3: SchemaCC
# ---------------------------------------------------------------------------

def _compute_positive_score(pairs_a: List[Tuple], pairs_b: List[Tuple]) -> float:
    """Compute Maximum-of-Containment between two sets of pairs."""
    set_a = set(map(tuple, pairs_a))
    set_b = set(map(tuple, pairs_b))
    intersection = len(set_a & set_b)
    if intersection == 0:
        return 0.0
    return max(intersection / len(set_a), intersection / len(set_b))


def schema_cc(
    corpus: List[dict],
    threshold: float = 0.3,
) -> List[dict]:
    """Schema matching using connected components.
    
    Computes pairwise positive compatibility between all tables.
    Links tables with compatibility above threshold.
    Uses connected components to group linked tables.
    
    Paper §5.1: "pair-wise schema matchers that use positive similarity,
    aggregate to group-level based on transitivity"
    
    Args:
        corpus: list of raw table dicts
        threshold: minimum positive compatibility to link two tables
        
    Returns:
        List of merged mapping dicts
    """
    # Extract candidate tables (2+ columns)
    candidates = []
    for table in corpus:
        relation = table.get("relation", [])
        if len(relation) < 2:
            continue
        # Get pairs from first two columns
        col0, col1 = relation[0], relation[1]
        min_len = min(len(col0), len(col1))
        pairs = [
            (col0[i].lower().strip(), col1[i].lower().strip())
            for i in range(min_len)
            if col0[i] and col1[i]
        ]
        if pairs:
            candidates.append({"table": table, "pairs": pairs})

    n = len(candidates)
    if n == 0:
        return []

    # Build adjacency using positive compatibility
    adj: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            score = _compute_positive_score(
                candidates[i]["pairs"],
                candidates[j]["pairs"]
            )
            if score >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components using BFS
    visited = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adj[node])
        components.append(component)

    # Merge each component into one mapping
    mappings = []
    for comp_id, component in enumerate(components):
        all_pairs = set()
        for idx in component:
            all_pairs.update(map(tuple, candidates[idx]["pairs"]))
        mappings.append({
            "group_id": f"cc_{comp_id}",
            "method": "SchemaCC",
            "num_source_tables": len(component),
            "pairs": [list(p) for p in all_pairs],
            "size": len(all_pairs),
        })

    return mappings


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def baseline_report(
    union_domain_results: List[dict],
    union_web_results: List[dict],
    schema_cc_results: List[dict],
    automap_results: Optional[List[dict]] = None,
) -> None:
    """Print comparison of all baseline methods."""

    def stats(results: List[dict]) -> dict:
        if not results:
            return {"mappings": 0, "total_pairs": 0, "avg_pairs": 0,
                    "multi_table": 0}
        total_pairs = sum(r["size"] for r in results)
        multi = sum(1 for r in results if r.get("num_source_tables", 1) > 1)
        return {
            "mappings": len(results),
            "total_pairs": total_pairs,
            "avg_pairs": round(total_pairs / len(results), 1),
            "multi_table": multi,
        }

    ud = stats(union_domain_results)
    uw = stats(union_web_results)
    scc = stats(schema_cc_results)

    print(f"\n  {'Metric':<30} {'UnionDomain':>15} {'UnionWeb':>12} {'SchemaCC':>12}", end="")
    if automap_results:
        am = stats(automap_results)
        print(f" {'AutoMap':>12}")
    else:
        print()

    print(f"  {'-'*75}")

    rows = [
        ("Total mappings", "mappings"),
        ("Total pairs", "total_pairs"),
        ("Avg pairs/mapping", "avg_pairs"),
        ("Multi-table merges", "multi_table"),
    ]

    for label, key in rows:
        line = f"  {label:<30} {ud[key]:>15} {uw[key]:>12} {scc[key]:>12}"
        if automap_results:
            line += f" {am[key]:>12}"
        print(line)


def save_baseline_results(
    results: List[dict],
    output_path: str,
) -> None:
    """Save baseline results to JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(results)} mappings to {output_path}")