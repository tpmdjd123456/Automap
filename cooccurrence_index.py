"""Build a global co-occurrence index over a table corpus.

For each column in each table:
- value_count[v] increments by 1 for every distinct value v in the column
  (set, not multiset — a value present 50 times in one column counts once).
- cooccurrence[(min(u,v), max(u,v))] increments by 1 for every distinct
  pair (u, v) of values in the column.
- total_columns increments by 1.

The resulting `Index = (cooccurrence, value_count, total_columns)` lets us
compute p(u), p(v), p(u,v) as the fractions used in PMI.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from tqdm import tqdm

from data_loader import Table
from noise_filter import is_noise_value

Index = Tuple[Dict[Tuple[str, str], int], Dict[str, int], int]


def build_cooccurrence_index(
    corpus: List[Table], skip_noise_values: bool = True,
) -> Index:
    """Single pass over the corpus building (cooccurrence, value_count, N).

    With ``skip_noise_values=True`` (default), pure-numeric / hex / placeholder
    tokens are excluded from the index — same predicate Vertica applies at the
    column level, now at the value level inside surviving mixed columns. Cuts
    index size 30-50% with near-zero NPMI signal loss.

    Set ``skip_noise_values=False`` for paper-strict behavior. Cache files built
    under different flag values are NOT interchangeable; main.py distinguishes
    them by path suffix.
    """
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    value_count: Dict[str, int] = defaultdict(int)
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
            for v in distinct:
                value_count[v] += 1
            for u, v in combinations(distinct, 2):
                cooccurrence[(u, v)] += 1
    return cooccurrence, value_count, total_columns


def save_index(index: Index, filepath: str) -> None:
    """Pickle the index. Pickle (not JSON) because tuple keys are first-class."""
    with open(filepath, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(filepath: str) -> Index:
    """Load a pickled index produced by `save_index`."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def index_summary(index: Index) -> None:
    """Print top-20 most common co-occurring pairs to stdout."""
    cooc, _vc, N = index
    top = sorted(cooc.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print(f"  Processed {N} columns")
    print(f"  Found {len(cooc)} unique value pairs")
    print("  Top co-occurring pairs:")
    for (u, v), count in top:
        print(f"    ({u}, {v}): {count}")
