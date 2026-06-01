"""Confidence Scoring — quality improvement for synthesized mappings.

Each mapping pair gets a confidence score between 0 and 1 based on:

1. Support count  — how many source tables contain this pair
2. Theta score    — average FD score of candidates contributing this pair
3. Conflict bonus — pairs that survived conflict resolution get a boost

Higher confidence = more reliable mapping pair.

This helps human curators focus on low-confidence pairs first,
and allows downstream applications like auto-join to use confidence
as a reliability signal.

Usage:
    from confidence_scorer import ConfidenceScorer
    scorer = ConfidenceScorer()
    scored = scorer.score_mappings(
        synthesized_path='output_1000/synthesized_mappings.jsonl',
        candidates_path='output_1000/candidates.jsonl',
        resolved_path='output_1000/resolved_mappings.jsonl',
    )
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Pair = Tuple[str, str]
PairScores = Dict[Pair, float]


# ---------------------------------------------------------------------------
# Core confidence scoring
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Computes confidence scores for mapping pairs.
    
    Combines three signals:
    - Support: how many source tables agree on this pair
    - Theta: average FD quality of contributing candidates
    - Conflict: whether pair survived conflict resolution
    """

    def __init__(
        self,
        weight_support: float = 0.5,
        weight_theta: float = 0.3,
        weight_conflict: float = 0.2,
        conflict_bonus: float = 1.0,
        conflict_penalty: float = 0.3,
    ):
        """Initialize confidence scorer with weights.
        
        Args:
            weight_support: weight for support count signal (0-1)
            weight_theta: weight for theta score signal (0-1)
            weight_conflict: weight for conflict survival signal (0-1)
            conflict_bonus: multiplier for pairs that survived resolution
            conflict_penalty: multiplier for pairs removed by resolution
        """
        assert abs(weight_support + weight_theta + weight_conflict - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.weight_support = weight_support
        self.weight_theta = weight_theta
        self.weight_conflict = weight_conflict
        self.conflict_bonus = conflict_bonus
        self.conflict_penalty = conflict_penalty

    def _count_pair_support(
        self,
        candidate_indices: List[int],
        candidates: List[dict],
    ) -> Dict[Pair, int]:
        """Count how many candidates support each pair."""
        support: Dict[Pair, int] = defaultdict(int)
        for idx in candidate_indices:
            if idx < len(candidates):
                for pair in candidates[idx]["pairs"]:
                    support[tuple(pair)] += 1
        return support

    def _get_pair_theta(
        self,
        candidate_indices: List[int],
        candidates: List[dict],
    ) -> Dict[Pair, float]:
        """Get average theta score for each pair across candidates."""
        theta_sum: Dict[Pair, float] = defaultdict(float)
        theta_count: Dict[Pair, int] = defaultdict(int)
        for idx in candidate_indices:
            if idx < len(candidates):
                theta = candidates[idx].get("theta", 1.0)
                for pair in candidates[idx]["pairs"]:
                    key = tuple(pair)
                    theta_sum[key] += theta
                    theta_count[key] += 1
        return {
            pair: theta_sum[pair] / theta_count[pair]
            for pair in theta_sum
        }

    def score_partition(
        self,
        synthesized: dict,
        candidates: List[dict],
        resolved_pairs: Optional[List[Pair]] = None,
    ) -> List[dict]:
        """Score all pairs in one synthesized mapping partition.
        
        Args:
            synthesized: one record from synthesized_mappings.jsonl
            candidates: all candidates from candidates.jsonl
            resolved_pairs: pairs kept after conflict resolution (optional)
            
        Returns:
            List of dicts with pair and confidence score
        """
        candidate_indices = synthesized.get("candidate_indices", [])
        pairs = [tuple(p) for p in synthesized["pairs"]]

        # Signal 1: Support count
        support = self._count_pair_support(candidate_indices, candidates)
        max_support = max(support.values()) if support else 1

        # Signal 2: Theta scores
        thetas = self._get_pair_theta(candidate_indices, candidates)

        # Signal 3: Conflict survival set
        resolved_set = set(resolved_pairs) if resolved_pairs else None

        scored_pairs = []
        for pair in pairs:
            # Normalize support to [0, 1]
            support_score = support.get(pair, 1) / max_support

            # Theta score already in [0.95, 1.0] — normalize to [0, 1]
            theta_score = (thetas.get(pair, 0.95) - 0.95) / 0.05
            theta_score = min(1.0, max(0.0, theta_score))

            # Conflict signal
            if resolved_set is not None:
                conflict_score = self.conflict_bonus if pair in resolved_set \
                    else self.conflict_penalty
            else:
                conflict_score = 1.0

            # Weighted combination
            confidence = (
                self.weight_support * support_score +
                self.weight_theta * theta_score +
                self.weight_conflict * conflict_score
            )

            # Clip to [0, 1]
            confidence = min(1.0, max(0.0, confidence))

            scored_pairs.append({
                "pair": list(pair),
                "confidence": round(confidence, 4),
                "support_count": support.get(pair, 1),
                "avg_theta": round(thetas.get(pair, 0.95), 4),
                "survived_conflict": resolved_set is None or pair in resolved_set,
            })

        # Sort by confidence descending
        scored_pairs.sort(key=lambda x: x["confidence"], reverse=True)
        return scored_pairs

    def score_mappings(
        self,
        synthesized_path: str,
        candidates_path: str,
        resolved_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> List[dict]:
        """Score all mappings in the corpus.
        
        Args:
            synthesized_path: path to synthesized_mappings.jsonl
            candidates_path: path to candidates.jsonl
            resolved_path: optional path to resolved_mappings.jsonl
            output_path: optional path to save scored mappings
            
        Returns:
            List of scored mapping dicts
        """
        # Load data
        synthesized = []
        with open(synthesized_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    synthesized.append(json.loads(line))

        candidates = []
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["pairs"] = [tuple(p) for p in rec["pairs"]]
                    candidates.append(rec)

        # Load resolved mappings if provided
        resolved_lookup: Dict[int, List[Pair]] = {}
        if resolved_path and os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        resolved_lookup[rec["partition_id"]] = [
                            tuple(p) for p in rec["pairs"]
                        ]

        print(f"  Loaded {len(synthesized)} synthesized mappings")
        print(f"  Loaded {len(candidates)} candidates")
        if resolved_lookup:
            print(f"  Loaded {len(resolved_lookup)} resolved mappings")

        # Score all partitions
        results = []
        for rec in synthesized:
            pid = rec["partition_id"]
            resolved_pairs = resolved_lookup.get(pid)
            scored_pairs = self.score_partition(rec, candidates, resolved_pairs)

            avg_conf = sum(p["confidence"] for p in scored_pairs) / len(scored_pairs) \
                if scored_pairs else 0.0

            results.append({
                "partition_id": pid,
                "size": len(scored_pairs),
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(min(p["confidence"] for p in scored_pairs), 4) \
                    if scored_pairs else 0.0,
                "max_confidence": round(max(p["confidence"] for p in scored_pairs), 4) \
                    if scored_pairs else 0.0,
                "scored_pairs": scored_pairs,
            })

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for rec in results:
                    f.write(json.dumps(rec) + "\n")
            print(f"  Saved scored mappings to {output_path}")

        return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def confidence_report(results: List[dict]) -> None:
    """Print a summary of confidence scores."""
    all_confs = [p["confidence"] for r in results for p in r["scored_pairs"]]
    if not all_confs:
        print("  No pairs to report.")
        return

    avg = sum(all_confs) / len(all_confs)
    high = sum(1 for c in all_confs if c >= 0.8)
    medium = sum(1 for c in all_confs if 0.5 <= c < 0.8)
    low = sum(1 for c in all_confs if c < 0.5)

    print(f"\n  Confidence Score Report:")
    print(f"  Total pairs scored    : {len(all_confs)}")
    print(f"  Average confidence    : {avg:.3f}")
    print(f"  High confidence (≥0.8): {high} ({100*high//len(all_confs)}%)")
    print(f"  Medium (0.5-0.8)      : {medium} ({100*medium//len(all_confs)}%)")
    print(f"  Low confidence (<0.5) : {low} ({100*low//len(all_confs)}%)")

    # Show top 5 most confident mappings
    top = sorted(results, key=lambda x: x["avg_confidence"], reverse=True)[:5]
    print(f"\n  Top 5 most confident mappings:")
    for r in top:
        sample = r["scored_pairs"][0]["pair"] if r["scored_pairs"] else []
        print(
            f"    partition {r['partition_id']}: "
            f"avg={r['avg_confidence']:.3f} "
            f"size={r['size']} "
            f"sample={sample}"
        )

    # Show bottom 5 least confident
    bottom = sorted(results, key=lambda x: x["avg_confidence"])[:5]
    print(f"\n  Bottom 5 least confident mappings:")
    for r in bottom:
        sample = r["scored_pairs"][0]["pair"] if r["scored_pairs"] else []
        print(
            f"    partition {r['partition_id']}: "
            f"avg={r['avg_confidence']:.3f} "
            f"size={r['size']} "
            f"sample={sample}"
        )