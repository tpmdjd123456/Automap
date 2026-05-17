"""PMI / NPMI / coherence over a co-occurrence index.

Per Wang & He (SIGMOD 2017), Section 3.1:

    PMI(u, v)  = log( p(u, v) / ( p(u) * p(v) ) )
    NPMI(u, v) = PMI(u, v) / -log( p(u, v) )      ∈ [-1, +1]

with p(x) = |C(x)| / N and p(u, v) = |C(u) ∩ C(v)| / N over a corpus of
N columns. NPMI is clipped to [-1, +1] to absorb floating-point drift.

Edge cases:
- Pair never co-occurs   -> NPMI = -1.0
- Either value not in corpus -> NPMI = -1.0
- p(u, v) = 1 (in every column) -> NPMI = +1.0 (-log p = 0 guarded)
- Any math domain error  -> NPMI = -1.0
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Tuple

from cooccurrence_index import Index
from data_loader import Table

ScoredColumn = Tuple[int, int, List[str], float]


def _pair_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def compute_pmi(u: str, v: str, index: Index) -> float:
    """PMI of (u, v). Returns -inf when any probability is zero.

    Most callers should use `compute_npmi` instead; PMI alone is unbounded.
    """
    cooc, vc, N = index
    co = cooc.get(_pair_key(u, v), 0)
    cu = vc.get(u, 0)
    cv = vc.get(v, 0)
    if co == 0 or cu == 0 or cv == 0 or N == 0:
        return float("-inf")
    p_uv = co / N
    p_u = cu / N
    p_v = cv / N
    return math.log(p_uv / (p_u * p_v))


def compute_npmi(u: str, v: str, index: Index) -> float:
    """NPMI of (u, v), clipped to [-1, +1]."""
    cooc, vc, N = index
    co = cooc.get(_pair_key(u, v), 0)
    cu = vc.get(u, 0)
    cv = vc.get(v, 0)
    if co == 0 or cu == 0 or cv == 0 or N == 0:
        return -1.0
    p_uv = co / N
    if p_uv >= 1.0:
        return 1.0
    try:
        pmi = math.log(p_uv / ((cu / N) * (cv / N)))
        npmi = pmi / -math.log(p_uv)
    except (ValueError, ZeroDivisionError):
        return -1.0
    if math.isnan(npmi) or math.isinf(npmi):
        return -1.0
    return max(-1.0, min(1.0, npmi))


def compute_coherence(column: List[str], index: Index) -> float:
    """Mean NPMI over all distinct unordered value pairs of `column`.

    Returns -1.0 if the column has fewer than 2 unique values (degenerate;
    the loader normally filters these out, but compute_coherence is safe to
    call directly).
    """
    distinct = sorted(v for v in set(column) if v)
    if len(distinct) < 2:
        return -1.0
    total = 0.0
    n = 0
    for u, v in combinations(distinct, 2):
        total += compute_npmi(u, v, index)
        n += 1
    return total / n


def score_corpus(corpus: List[Table], index: Index) -> List[ScoredColumn]:
    """Score every column in the corpus.

    Returns a list of `(table_idx, col_idx, column_values, coherence_score)`
    tuples. `col_idx` is the column's position within the (already-loaded
    and pre-filtered) table, not its original position in `relation`.
    """
    out: List[ScoredColumn] = []
    for ti, (_metadata, columns) in enumerate(corpus):
        for ci, col in enumerate(columns):
            out.append((ti, ci, col, compute_coherence(col, index)))
    return out


def test_npmi(index: Index) -> None:
    """Print sanity stats for an index. Verifies symmetry and range on a
    sample of pairs and prints the highest-ranked pair. Helper for `main.py`,
    not a pytest test."""
    cooc, _vc, _N = index
    if not cooc:
        print("  NPMI sanity: index is empty")
        return
    # Top pair by raw co-occurrence.
    top_pair, top_count = max(cooc.items(), key=lambda kv: kv[1])
    u, v = top_pair
    print(f"  NPMI sanity: top pair ({u}, {v}) count={top_count} npmi={compute_npmi(u, v, index):.3f}")
    # Symmetry on first 100 pairs.
    sym_ok = all(
        compute_npmi(u, v, index) == compute_npmi(v, u, index)
        for u, v in list(cooc.keys())[:100]
    )
    print(f"  NPMI sanity: symmetric on first 100 pairs: {sym_ok}")
    # Range check on first 200 pairs.
    in_range = all(
        -1.0 <= compute_npmi(u, v, index) <= 1.0
        for u, v in list(cooc.keys())[:200]
    )
    print(f"  NPMI sanity: in-range on first 200 pairs: {in_range}")
