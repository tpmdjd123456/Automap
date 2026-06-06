"""Build a global co-occurrence index over a table corpus.

For each column in each table:
- value_count[id] increments by 1 for every distinct value in the column
  (set, not multiset).
- cooccurrence[pack(min(uid,vid), max(uid,vid))] increments by 1 for every
  distinct pair.
- total_columns increments by 1.

Values are *interned*: each unique string is assigned a 32-bit integer ID via
``str_to_id``. The cooccurrence dict stores packed int64 keys
``(min_id << 32) | max_id`` instead of string tuples — saves ~24 % per entry at
billion-pair scale (Python tuple / str-pointer overhead is the dominant cost).

The resulting ``Index = (cooc, value_count, total_columns, str_to_id)`` lets us
compute p(u), p(v), p(u,v) as the fractions used in PMI. ``str_to_id`` is the
build-time string→ID map; lookups need it to translate query strings.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from tqdm import tqdm

from data_loader import Table
from noise_filter import is_noise_value

# Index now carries the string→ID map so external lookups can translate
Index = Tuple[Dict[int, int], Dict[int, int], int, Dict[str, int]]


def pack(a: int, b: int) -> int:
    """Pack two non-negative int IDs (each <2**32) into one int64.

    Caller MUST ensure ``a <= b`` (we order on the build side; lookup callers
    must order too). Encoding kept positive so it stays in a Python int.
    """
    return (a << 32) | b


def build_cooccurrence_index(
    corpus: List[Table], skip_noise_values: bool = True,
) -> Index:
    """Single pass over the corpus building
    (cooccurrence, value_count, N, str_to_id).

    With ``skip_noise_values=True`` (default), pure-numeric / hex / placeholder
    tokens are excluded from the index — same predicate Vertica applies at the
    column level, now at the value level inside surviving mixed columns.

    Set ``skip_noise_values=False`` for paper-strict behavior. Cache files built
    under different flag values are NOT interchangeable; main.py distinguishes
    them by path suffix.
    """
    str_to_id: Dict[str, int] = {}
    cooccurrence: Dict[int, int] = defaultdict(int)
    value_count: Dict[int, int] = defaultdict(int)
    total_columns = 0
    for _metadata, columns in tqdm(
        corpus, desc="Building co-occurrence index", unit="table",
        mininterval=30.0,
    ):
        for col in columns:
            if skip_noise_values:
                distinct = sorted(
                    v for v in set(col) if v and not is_noise_value(v)
                )
            else:
                distinct = sorted(v for v in set(col) if v)
            if len(distinct) < 2:
                continue
            total_columns += 1
            # Intern: assign IDs to first-seen values; reuse on later sightings.
            distinct_ids: List[int] = []
            for s in distinct:
                sid = str_to_id.get(s)
                if sid is None:
                    sid = len(str_to_id)
                    str_to_id[s] = sid
                distinct_ids.append(sid)
            for sid in distinct_ids:
                value_count[sid] += 1
            for a, b in combinations(distinct_ids, 2):
                # combinations() preserves order; distinct was sorted by
                # *string*, not by ID, so we need to canonicalize here.
                if a > b:
                    a, b = b, a
                cooccurrence[pack(a, b)] += 1
    return cooccurrence, value_count, total_columns, str_to_id


def save_index(index: Index, filepath: str) -> None:
    """Pickle the index. Pickle (not JSON) because dict keys aren't JSON-able."""
    with open(filepath, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(filepath: str) -> Index:
    """Load a pickled index produced by ``save_index``."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def index_summary(index: Index) -> None:
    """Print top-20 most common co-occurring pairs to stdout."""
    cooc, _vc, N, str_to_id = index
    # Build reverse map only for the top-20 lookup (small, transient).
    top = sorted(cooc.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print(f"  Processed {N} columns")
    print(f"  Found {len(cooc)} unique value pairs")
    if not top:
        return
    id_to_str: Dict[int, str] = {sid: s for s, sid in str_to_id.items()}
    print("  Top co-occurring pairs:")
    for packed, count in top:
        a = packed >> 32
        b = packed & 0xFFFFFFFF
        print(f"    ({id_to_str[a]}, {id_to_str[b]}): {count}")
