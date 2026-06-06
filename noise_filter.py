"""Drop non-informational candidates before greedy_partition (WP3).

A candidate is a list of (left, right) pairs from one table's column-pair. We
drop the candidate when the pair set has no mapping content worth synthesizing
across tables. Rules are intentionally narrow — they target patterns that the
advisor's "no scenario where this would be useful" test rules out.

See `is_noise(pairs)` for the predicate. Returns (drop: bool, reason: str|None).
Reasons (for reporting):
  identity              - L == R for every pair
  int->int              - every left AND every right value is numeric
  rank->rank            - both columns are consecutive integer runs (1..n)
  hex_pair              - every value on both sides matches #hex or 0xhex
  placeholder_dominated - every pair has at least one side that is a
                          placeholder token (-, --, '', n/a, na, null, none, ?)
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# A "number" for our purposes: optional sign, digits, optional single decimal
# group with `.` or `,`. Deliberately strict — we don't want to match "$1.2m"
# or "-0.0%" as numeric (those still have signal as text).
_NUM = re.compile(r"^-?\d+(?:[.,]\d+)?$")
_INT = re.compile(r"^-?\d+$")
_HEX = re.compile(r"^(?:#|0x)[0-9a-fA-F]+$")

# Tokens that mean "no value" in web tables. `0` is *not* a placeholder — it's a
# real measurement in many contexts (counts, scores).
_PLACEHOLDERS = frozenset({"", "-", "--", "—", "n/a", "na", "null", "none", "?"})


def _is_num(v: str) -> bool:
    return bool(_NUM.match(v.strip()))


def _is_int(v: str) -> bool:
    return bool(_INT.match(v.strip()))


def _is_hex(v: str) -> bool:
    return bool(_HEX.match(v.strip()))


def _is_placeholder(v: str) -> bool:
    return v.strip().lower() in _PLACEHOLDERS


def _is_rank_run(vals: Sequence[str]) -> bool:
    """True iff `vals` are a consecutive integer sequence (any order)
    anchored at 0 or 1, e.g. {1,2,…,n} or {0,1,…,n-1}, length >= 3."""
    if len(vals) < 3 or not all(_is_int(v) for v in vals):
        return False
    nums = sorted(int(v) for v in vals)
    if len(set(nums)) != len(nums):
        return False
    return nums[0] in (0, 1) and nums == list(range(nums[0], nums[0] + len(nums)))


def is_noise(pairs: Sequence[Tuple[str, str]]) -> Tuple[bool, Optional[str]]:
    """Classify a candidate's pair list. Returns (drop, reason).

    Drop iff any rule matches; first matching rule's name is the reason.
    """
    if not pairs:
        return True, "empty"

    if all(l == r for l, r in pairs):
        return True, "identity"

    lefts = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]

    if all(_is_num(v) for v in lefts) and all(_is_num(v) for v in rights):
        return True, "int->int"

    if _is_rank_run(lefts) and _is_rank_run(rights):
        return True, "rank->rank"

    if all(_is_hex(v) for v in lefts) and all(_is_hex(v) for v in rights):
        return True, "hex_pair"

    if all(_is_placeholder(l) or _is_placeholder(r) for l, r in pairs):
        return True, "placeholder_dominated"

    return False, None


def filter_candidates(candidates: List[dict]) -> Tuple[List[dict], dict]:
    """Apply `is_noise` to a list of candidate dicts (each with a "pairs" key).
    Returns (kept_candidates, drop_counts_by_reason)."""
    kept: List[dict] = []
    drops: dict = {}
    for c in candidates:
        drop, reason = is_noise(c["pairs"])
        if drop:
            drops[reason] = drops.get(reason, 0) + 1
        else:
            kept.append(c)
    return kept, drops
