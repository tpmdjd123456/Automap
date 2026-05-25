"""Table Synthesis — WP3 (paper §4).

Implements the greedy partitioning algorithm (Algorithm 3) from Wang & He
(SIGMOD 2017) Section 4.2. Takes candidate two-column tables from WP2
(candidates.jsonl) and groups them into synthesized mapping relationships.

Pipeline:
  1. Load candidates (from candidates.jsonl produced by WP2)
  2. Build inverted index for efficiency
  3. Compute pairwise positive + negative compatibility scores
  4. Run greedy partitioning until no more compatible merges exist
  5. Union each partition into one synthesized mapping table
  6. Save synthesized_mappings.jsonl + print report
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Candidate = Dict[str, Any]
MappingTable = List[Tuple[str, str]]
Partition = List[int]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> List[Candidate]:
    """Load candidates.jsonl produced by WP2.

    Converts each ``pairs`` field from ``List[List[str]]`` (on disk) to
    ``List[Tuple[str, str]]`` in memory so the rest of WP3 can use tuple
    unpacking.
    """
    candidates: List[Candidate] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["pairs"] = [tuple(p) for p in rec["pairs"]]
            candidates.append(rec)
    return candidates


# ---------------------------------------------------------------------------
# Approximate string matching  (paper §4.1, Algorithm 2)
# ---------------------------------------------------------------------------

def _naive_edit_distance(v1: str, v2: str) -> int:
    """Full O(m*n) edit distance — used only as a reference / for threshold=0."""
    m, n = len(v1), len(v2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if v1[i - 1] == v2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def approx_match(v1: str, v2: str, fed: float = 0.2, ked: int = 10) -> bool:
    """Return True if ``edit_distance(v1, v2) <= threshold``.

    The threshold is computed per the paper formula::

        threshold = min(floor(len(v1) * fed), floor(len(v2) * fed), ked)

    Uses band dynamic programming (Ukkonen-style) when threshold > 0 —
    only cells within ``threshold`` columns of the diagonal are computed,
    giving O(threshold * min(len1, len2)) time instead of O(len1 * len2).

    Args:
        v1: first string.
        v2: second string.
        fed: fractional edit-distance factor (default 0.2 per paper).
        ked: absolute upper cap on threshold (default 10 per paper).

    Returns:
        True if the edit distance is within the threshold.
    """
    threshold = min(
        math.floor(len(v1) * fed),
        math.floor(len(v2) * fed),
        ked,
    )
    if v1 == v2:
        return True
    if threshold == 0:
        return False

    m, n = len(v1), len(v2)
    # Early-exit: length difference alone exceeds threshold
    if abs(m - n) > threshold:
        return False

    # Band DP — row i corresponds to v1[0..i-1]
    # We only track columns j in [max(0, i-threshold) .. min(n, i+threshold)]
    INF = threshold + 1
    # Use a flat array of size n+1, but only update within the band
    prev = list(range(n + 1))
    # Clamp initial row values outside the band to INF
    for j in range(threshold + 1, n + 1):
        prev[j] = INF

    for i in range(1, m + 1):
        curr = [INF] * (n + 1)
        curr[0] = i if i <= threshold else INF
        j_lo = max(1, i - threshold)
        j_hi = min(n, i + threshold)
        for j in range(j_lo, j_hi + 1):
            cost = 0 if v1[i - 1] == v2[j - 1] else 1
            del_cost = (prev[j] + 1) if prev[j] < INF else INF
            ins_cost = (curr[j - 1] + 1) if curr[j - 1] < INF else INF
            sub_cost = (prev[j - 1] + cost) if prev[j - 1] < INF else INF
            curr[j] = min(del_cost, ins_cost, sub_cost)
        prev = curr

    return prev[n] <= threshold


# ---------------------------------------------------------------------------
# Pairwise compatibility scores  (paper §4.1, Equations 3 & 4)
# ---------------------------------------------------------------------------

def _count_intersection(
    b: List[Tuple[str, str]],
    b_prime: List[Tuple[str, str]],
    use_approx: bool,
) -> int:
    """Count |B ∩ B'| — pairs that match across both tables."""
    count = 0
    for l1, r1 in b:
        for l2, r2 in b_prime:
            if use_approx:
                if approx_match(l1, l2) and approx_match(r1, r2):
                    count += 1
                    break
            else:
                if l1 == l2 and r1 == r2:
                    count += 1
                    break
    return count


def positive_score(
    b: List[Tuple[str, str]],
    b_prime: List[Tuple[str, str]],
    use_approx: bool = True,
) -> float:
    """Maximum-of-Containment similarity (Equation 3 from paper).

    ``w+(B, B') = max( |B ∩ B'| / |B| , |B ∩ B'| / |B'| )``

    Returns a value in [0, 1].  When ``use_approx=True``, pair matching
    uses approximate string comparison via :func:`approx_match`.
    """
    if not b or not b_prime:
        return 0.0
    intersection = _count_intersection(b, b_prime, use_approx)
    if intersection == 0:
        return 0.0
    return max(intersection / len(b), intersection / len(b_prime))


def _conflict_set(
    b: List[Tuple[str, str]],
    b_prime: List[Tuple[str, str]],
    use_approx: bool,
) -> Set[str]:
    """Compute F(B, B') — left values where B and B' disagree on the right.

    F = {l | (l, r) in B and (l, r') in B' and r != r' (approx)}
    """
    # Build mapping: left -> set of right values in b_prime (approx-matched)
    b_prime_left: Dict[str, List[str]] = defaultdict(list)
    for l2, r2 in b_prime:
        b_prime_left[l2].append(r2)

    conflicts: Set[str] = set()
    for l1, r1 in b:
        # Find matching left values in b_prime
        for l2, rights in b_prime_left.items():
            left_matches = approx_match(l1, l2) if use_approx else (l1 == l2)
            if left_matches:
                for r2 in rights:
                    # Conflict: left values match but right values differ (not approx-equal)
                    right_differs = not (approx_match(r1, r2) if use_approx else (r1 == r2))
                    if right_differs:
                        conflicts.add(l1)
                        break
    return conflicts


def negative_score(
    b: List[Tuple[str, str]],
    b_prime: List[Tuple[str, str]],
    use_approx: bool = True,
) -> float:
    """Negative incompatibility score (Equation 4 from paper).

    ``w-(B, B') = -max( |F| / |B| , |F| / |B'| )``

    where ``F = {l | (l,r) in B and (l,r') in B' and r != r'}``.

    Returns a value in [-1, 0].
    """
    if not b or not b_prime:
        return 0.0
    # Compute conflict from both directions (symmetrised)
    f_from_b = _conflict_set(b, b_prime, use_approx)
    f_from_b_prime = _conflict_set(b_prime, b, use_approx)
    # F(B, B') from b's perspective uses left values of b
    # |F| is counted once per distinct conflicting left value in b
    # and separately in b' — take max of both ratios
    size = max(len(f_from_b) / len(b), len(f_from_b_prime) / len(b_prime))
    return -size


# ---------------------------------------------------------------------------
# Inverted index  (paper §4.1)
# ---------------------------------------------------------------------------

def build_inverted_index(
    candidates: List[Candidate],
) -> Tuple[Dict[Tuple[str, str], List[int]], Dict[str, List[int]]]:
    """Build two inverted indexes for efficient candidate-pair lookup.

    Returns:
        pair_index: maps ``(l, r)`` → list of candidate indices that
            contain that exact pair.
        left_index: maps ``l`` → list of candidate indices that have
            ``l`` as a left-hand value.
    """
    pair_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    left_index: Dict[str, List[int]] = defaultdict(list)
    for idx, cand in enumerate(candidates):
        seen_left: Set[str] = set()
        for pair in cand["pairs"]:
            l, r = pair[0], pair[1]
            pair_index[(l, r)].append(idx)
            if l not in seen_left:
                left_index[l].append(idx)
                seen_left.add(l)
    return dict(pair_index), dict(left_index)


# ---------------------------------------------------------------------------
# Greedy partitioning  (paper §4.2, Algorithm 3)
# ---------------------------------------------------------------------------
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Iterable, Set, Optional

def greedy_partition(
    candidates: List[Candidate],
    tau: float = -0.2,
    theta_overlap: int = 1,
    use_approx: bool = True,
    output_folder: str = "output"
) -> List[Partition]:
    """Run greedy table synthesis (Algorithm 3 from the paper).

    Args:
        candidates: list of candidate dicts from WP2 (candidates.jsonl).
        tau: negative-weight threshold; merges where ``w- < tau`` are
            blocked (default -0.2 per paper §4.2).
        theta_overlap: minimum shared value pairs to consider a candidate
            pair at all (efficiency filter, default 1).
        use_approx: whether to use approximate string matching.

    Returns:
        List of partitions.  Each partition is a list of candidate indices
        (into the input ``candidates`` list) that belong to the same
        synthesized mapping relationship.  Every candidate appears in
        exactly one partition.
    """
    n = len(candidates)
    if n == 0:
        return []

    print(f"  Building inverted index...")
    pair_index, left_index = build_inverted_index(candidates)
    print(f"    pair_index: {len(pair_index)} unique pairs")
    print(f"    left_index: {len(left_index)} unique left values")

    # ------------------------------------------------------------------
    # Each partition is identified by an integer ID.
    # part_members[pid] = set of candidate indices in that partition
    # part_id[cand_idx] = which partition id contains this candidate
    # ------------------------------------------------------------------
    part_members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    # For score computation we need the union of pairs for each partition
    part_pairs: Dict[int, List[Tuple[str, str]]] = {
        i: list(candidates[i]["pairs"]) for i in range(n)
    }
    next_pid = n  # monotonically increasing partition IDs

    # ------------------------------------------------------------------
    # Compute initial pairwise scores using the inverted index
    # pos_scores[(a,b)] = w+   (a < b by convention)
    # neg_scores[(a,b)] = w-
    # ------------------------------------------------------------------
    print(f"  Computing initial compatibility graph...")

    # Find candidate pairs with overlap via inverted index
    # pair (ci, cj) where ci < cj
    overlapping_pairs: Set[Tuple[int, int]] = set()
    for indices in pair_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = indices[x], indices[y]
                if a > b:
                    a, b = b, a
                overlapping_pairs.add((a, b))
    for indices in left_index.values():
        if len(indices) < 2:
            continue
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                a, b = indices[x], indices[y]
                if a > b:
                    a, b = b, a
                overlapping_pairs.add((a, b))

    # partition-pair score tables: key = frozenset({pid1, pid2})
    # We use tuple (min, max) for ordered key
    pos_scores: Dict[Tuple[int, int], float] = {}
    neg_scores: Dict[Tuple[int, int], float] = {}

    positive_edges = 0
    blocking_edges = 0
    
    # --- ADDED TQDM HERE TO TRACK INITIAL SCORE COMPUTATION ---
    for ci, cj in tqdm(overlapping_pairs, desc="Calculating initial edge weights", unit="pair"):
        pi, pj = ci, cj  # initially pid == cand idx
        key = (pi, pj)
        bp = list(candidates[ci]["pairs"])
        bq = list(candidates[cj]["pairs"])
        ps = positive_score(bp, bq, use_approx=use_approx)
        ns = negative_score(bp, bq, use_approx=use_approx)
        if ps > 0:
            pos_scores[key] = ps
            positive_edges += 1
        if ns < 0:
            neg_scores[key] = ns
            if ns < tau:
                blocking_edges += 1

# --- SAVE EDGE SCORES TO FILE ---
    import json
    import os

    # Ensure the directory exists dynamically
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    scores_output_path = os.path.join(output_folder, "computed_edge_scores.jsonl")
    print(f"  Saving computed edge weights to {os.path.abspath(scores_output_path)}...")
    
    all_edge_keys = set(pos_scores.keys()).union(set(neg_scores.keys()))
    
    with open(scores_output_path, "w", encoding="utf-8") as f:
        for ci, cj in all_edge_keys:
            edge_data = {
                "cand_i": ci,
                "cand_j": cj,
                "w_pos": pos_scores.get((ci, cj), 0.0),
                "w_neg": neg_scores.get((ci, cj), 0.0)
            }
            f.write(json.dumps(edge_data) + "\n")
            
    print(f"  Successfully saved {len(all_edge_keys)} edge scores to {output_folder}/")

    print(f"    Non-zero positive edges: {positive_edges}")
    print(f"    Blocking negative edges (w- < tau): {blocking_edges}")
    print(f"  Running greedy partitioning...")

    merge_count = 0

    # --- ADDED TQDM HERE TO TRACK ITERATIVE MATRIX MERGING ---
    pbar_merge = tqdm(desc="Merging partitions", unit="round")
    while True:
        # Find the best merge: highest w+(P1,P2) where w-(P1,P2) >= tau
        best_key: Optional[Tuple[int, int]] = None
        best_pos: float = -1.0

        for key, ps in pos_scores.items():
            if ps <= best_pos:
                continue
            pi, pj = key
            if pi not in part_members or pj not in part_members:
                continue
            ns = neg_scores.get(key, 0.0)
            if ns >= tau:
                best_key = key
                best_pos = ps

        if best_key is None:
            break

        pi, pj = best_key
        ns_best = neg_scores.get(best_key, 0.0)
        size_i = len(part_members[pi])
        size_j = len(part_members[pj])
        merge_count += 1
        
        # Update progress bar statistics in real-time
        pbar_merge.update(1)
        pbar_merge.set_postfix({"last_w+": f"{best_pos:.3f}", "active_parts": len(part_members)})

        # Create new partition
        new_pid = next_pid
        next_pid += 1
        new_members = part_members[pi] + part_members[pj]
        # Union of pairs (deduplicated for scoring; synthesis deduplicates separately)
        seen: Set[Tuple[str, str]] = set()
        new_pairs: List[Tuple[str, str]] = []
        for pair in part_pairs[pi] + part_pairs[pj]:
            pt = (pair[0], pair[1])
            if pt not in seen:
                seen.add(pt)
                new_pairs.append(pt)

        part_members[new_pid] = new_members
        part_pairs[new_pid] = new_pairs

        # Update scores for new_pid vs all remaining partitions
        remaining = [
            pid for pid in part_members
            if pid != pi and pid != pj and pid != new_pid
        ]
        for pk in remaining:
            key_ik = (min(pi, pk), max(pi, pk))
            key_jk = (min(pj, pk), max(pj, pk))
            key_nk = (min(new_pid, pk), max(new_pid, pk))

            old_pos_ik = pos_scores.get(key_ik, 0.0)
            old_pos_jk = pos_scores.get(key_jk, 0.0)
            new_pos = old_pos_ik + old_pos_jk

            old_neg_ik = neg_scores.get(key_ik, 0.0)
            old_neg_jk = neg_scores.get(key_jk, 0.0)
            new_neg = min(old_neg_ik, old_neg_jk)

            if new_pos > 0:
                pos_scores[key_nk] = new_pos
            if new_neg < 0:
                neg_scores[key_nk] = new_neg

            # Clean up old keys
            pos_scores.pop(key_ik, None)
            pos_scores.pop(key_jk, None)
            neg_scores.pop(key_ik, None)
            neg_scores.pop(key_jk, None)

        # Remove old partition entries and the merged key
        del part_members[pi]
        del part_members[pj]
        del part_pairs[pi]
        del part_pairs[pj]
        pos_scores.pop(best_key, None)
        neg_scores.pop(best_key, None)

    pbar_merge.close()
    print(f"    Converged after {merge_count} merges. {len(part_members)} partitions.")
    return [sorted(members) for members in part_members.values()]


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def synthesize_mapping(
    partition: Partition,
    candidates: List[Candidate],
) -> MappingTable:
    """Union all candidates in a partition into one deduplicated mapping table.

    Returns a sorted list of unique ``(left, right)`` value pairs.
    """
    seen: Set[Tuple[str, str]] = set()
    result: List[Tuple[str, str]] = []
    for idx in partition:
        for pair in candidates[idx]["pairs"]:
            pt: Tuple[str, str] = (pair[0], pair[1])
            if pt not in seen:
                seen.add(pt)
                result.append(pt)
    return sorted(result)


def save_synthesized_mappings(
    partitions: List[Partition],
    candidates: List[Candidate],
    output_path: str,
) -> None:
    """Save synthesized mappings as JSONL. One line per partition.

    Each line has the schema::

        {
            "partition_id": int,
            "candidate_indices": [int, ...],
            "pairs": [[left, right], ...],
            "size": int,
            "num_source_tables": int
        }
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for pid, partition in enumerate(partitions):
            mapping = synthesize_mapping(partition, candidates)
            record = {
                "partition_id": pid,
                "candidate_indices": partition,
                "pairs": [list(p) for p in mapping],
                "size": len(mapping),
                "num_source_tables": len(partition),
            }
            fh.write(json.dumps(record) + "\n")


def synthesis_report(
    partitions: List[Partition],
    candidates: List[Candidate],
) -> None:
    """Print a human-readable summary of the synthesized mappings.

    Reports:
    - Total partitions produced
    - Singleton count (partitions with a single candidate)
    - Size distribution (min / mean / max pairs per partition)
    - Top 5 largest partitions with sample pairs
    """
    if not partitions:
        print("  No partitions produced.")
        return

    sizes = [len(synthesize_mapping(p, candidates)) for p in partitions]
    singletons = sum(1 for p in partitions if len(p) == 1)
    total = len(partitions)
    mean_size = sum(sizes) / total if total else 0.0

    print(f"  Synthesis report:")
    print(f"    Total synthesized mappings: {total}")
    print(f"    Singleton partitions: {singletons} ({100.0 * singletons / total:.1f}%)")
    print(f"    Multi-table partitions: {total - singletons}")
    print(
        f"    Pairs per mapping: min={min(sizes)} "
        f"mean={mean_size:.1f} max={max(sizes)}"
    )
    print(f"    Top 5 largest synthesized mappings:")
    indexed = sorted(enumerate(partitions), key=lambda x: sizes[x[0]], reverse=True)
    for rank, (i, partition) in enumerate(indexed[:5]):
        mapping = synthesize_mapping(partition, candidates)
        sample = mapping[:3]
        print(
            f"      [{sizes[i]} pairs, {len(partition)} source tables] "
            f"sample: {', '.join(f'({l}, {r})' for l, r in sample)}"
            + (" ..." if len(mapping) > 3 else "")
        )
