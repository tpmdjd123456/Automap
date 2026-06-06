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

The cooccurrence index is interned: lookups translate query strings to int IDs
via ``str_to_id`` carried in the index tuple, then key the packed-int dicts.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Tuple

from cooccurrence_index import Index, pack
from data_loader import Table

ScoredColumn = Tuple[int, int, List[str], float]


def compute_pmi(u: str, v: str, index: Index) -> float:
    """PMI of (u, v). Returns -inf when any probability is zero.

    Most callers should use `compute_npmi` instead; PMI alone is unbounded.
    """
    cooc, vc, N, s2i = index
    uid = s2i.get(u)
    vid = s2i.get(v)
    if uid is None or vid is None or N == 0:
        return float("-inf")
    a, b = (uid, vid) if uid <= vid else (vid, uid)
    co = cooc.get(pack(a, b), 0)
    cu = vc.get(uid, 0)
    cv = vc.get(vid, 0)
    if co == 0 or cu == 0 or cv == 0:
        return float("-inf")
    p_uv = co / N
    p_u = cu / N
    p_v = cv / N
    return math.log(p_uv / (p_u * p_v))


def compute_npmi(u: str, v: str, index: Index) -> float:
    """NPMI of (u, v), clipped to [-1, +1]."""
    cooc, vc, N, s2i = index
    uid = s2i.get(u)
    vid = s2i.get(v)
    if uid is None or vid is None or N == 0:
        return -1.0
    a, b = (uid, vid) if uid <= vid else (vid, uid)
    co = cooc.get(pack(a, b), 0)
    cu = vc.get(uid, 0)
    cv = vc.get(vid, 0)
    if co == 0 or cu == 0 or cv == 0:
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


def compute_coherence(
    column: List[str], index: Index, skip_noise_values: bool = True,
) -> float:
    """Mean NPMI over all distinct unordered value pairs of `column`.

    With ``skip_noise_values=True`` (default), pure-numeric / hex / placeholder
    tokens are excluded — must match how the index was built (see
    ``build_cooccurrence_index``). Mismatch produces silently-wrong scores
    because lookups for filtered values return NPMI=-1.

    Returns -1.0 if the column has fewer than 2 indexable unique values.

    Hot-path implementation: translate the column's distinct values to int IDs
    *once*, then iterate ID pairs against the packed index. Saves an
    str_to_id.get() per pair-loop iteration vs calling compute_npmi.
    """
    from noise_filter import is_noise_value
    if skip_noise_values:
        distinct = sorted(
            v for v in set(column) if v and not is_noise_value(v)
        )
    else:
        distinct = sorted(v for v in set(column) if v)
    if len(distinct) < 2:
        return -1.0
    cooc, vc, N, s2i = index
    ids = [s2i.get(v) for v in distinct]
    total = 0.0
    n = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            n += 1
            ui, vi = ids[i], ids[j]
            if ui is None or vi is None or N == 0:
                total += -1.0
                continue
            a, b = (ui, vi) if ui <= vi else (vi, ui)
            co = cooc.get(pack(a, b), 0)
            cu = vc.get(ui, 0)
            cv = vc.get(vi, 0)
            if co == 0 or cu == 0 or cv == 0:
                total += -1.0
                continue
            p_uv = co / N
            if p_uv >= 1.0:
                total += 1.0
                continue
            try:
                pmi = math.log(p_uv / ((cu / N) * (cv / N)))
                npmi = pmi / -math.log(p_uv)
            except (ValueError, ZeroDivisionError):
                total += -1.0
                continue
            if math.isnan(npmi) or math.isinf(npmi):
                total += -1.0
                continue
            total += max(-1.0, min(1.0, npmi))
    return total / n


def score_corpus(
    corpus: List[Table], index: Index, skip_noise_values: bool = True,
) -> List[ScoredColumn]:
    """Score every column in the corpus.

    ``skip_noise_values`` must match the flag used at index build time.

    Returns a list of `(table_idx, col_idx, column_values, coherence_score)`
    tuples. `col_idx` is the column's position within the (already-loaded
    and pre-filtered) table, not its original position in `relation`.
    """
    from tqdm import tqdm
    out: List[ScoredColumn] = []
    for ti, (_metadata, columns) in enumerate(tqdm(
        corpus, desc="Scoring coherence", unit="table", mininterval=30.0,
    )):
        for ci, col in enumerate(columns):
            out.append((ti, ci, col, compute_coherence(
                col, index, skip_noise_values=skip_noise_values,
            )))
    return out


def test_npmi(index: Index) -> None:
    """Print sanity stats for an index. Verifies symmetry and range on a
    sample of pairs and prints the highest-ranked pair. Helper for `main.py`,
    not a pytest test."""
    cooc, _vc, _N, s2i = index
    if not cooc:
        print("  NPMI sanity: index is empty")
        return
    # Top pair by raw co-occurrence (packed key → strings via reverse map).
    top_packed, top_count = max(cooc.items(), key=lambda kv: kv[1])
    id_to_str = {sid: s for s, sid in s2i.items()}
    a = top_packed >> 32
    b = top_packed & 0xFFFFFFFF
    u, v = id_to_str[a], id_to_str[b]
    print(
        f"  NPMI sanity: top pair ({u}, {v}) count={top_count} "
        f"npmi={compute_npmi(u, v, index):.3f}"
    )
    # Symmetry on first 100 pairs.
    sample_keys = list(cooc.keys())[:100]
    sym_ok = True
    for packed in sample_keys:
        a, b = packed >> 32, packed & 0xFFFFFFFF
        su, sv = id_to_str[a], id_to_str[b]
        if compute_npmi(su, sv, index) != compute_npmi(sv, su, index):
            sym_ok = False
            break
    print(f"  NPMI sanity: symmetric on first 100 pairs: {sym_ok}")
    # Range check on first 200 pairs.
    in_range = True
    for packed in list(cooc.keys())[:200]:
        a, b = packed >> 32, packed & 0xFFFFFFFF
        s = compute_npmi(id_to_str[a], id_to_str[b], index)
        if not (-1.0 <= s <= 1.0):
            in_range = False
            break
    print(f"  NPMI sanity: in-range on first 200 pairs: {in_range}")
