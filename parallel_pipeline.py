"""Parallel Pipeline — performance improvement for WP1 and WP2.

Parallelizes the most expensive stages of the pipeline:
- Stage 3: PMI coherence scoring (one worker per table)
- Stage 5: FD filtering (one worker per table)

Uses Python's multiprocessing.Pool for CPU-bound tasks.

Usage:
    from parallel_pipeline import parallel_score_corpus, parallel_fd_filter
    
    # Instead of: scored = score_corpus(corpus, index)
    scored = parallel_score_corpus(corpus, index, n_workers=4)
    
    # Instead of: candidates = fd_filter(filtered)
    candidates = parallel_fd_filter(filtered, index, theta=0.95)
"""

from __future__ import annotations

import time
import multiprocessing as mp
from typing import List, Tuple, Optional

from npmi import compute_coherence
from fd_filter import compute_approx_fd
from synthesis import positive_score, negative_score


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _score_table_worker(args: Tuple) -> List[Tuple]:
    """Worker function for scoring one table's columns.
    
    Args:
        args: tuple of (table_idx, table, index)
        
    Returns:
        List of (table_idx, col_idx, values, score) tuples
    """
    table_idx, table, index = args
    results = []
    metadata, columns = table
    for col_idx, column in enumerate(columns):
        score = compute_coherence(column, index)
        results.append((table_idx, col_idx, column, score))
    return results


def _fd_filter_worker(args: Tuple) -> List[dict]:
    """Worker function for FD filtering one table.
    
    Handles both tuple format (metadata, columns) and 
    dict format from JSONL files.
    """
    table_idx, table, theta, min_rows = args

    # Handle both tuple and dict formats
    if isinstance(table, dict):
        columns = table.get("relation", [])
        metadata = {k: v for k, v in table.items() 
                   if k not in {"relation", "coherence_scores", "rejected_column_indices"}}
    else:
        metadata, columns = table

    candidates = []
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i == j:
                continue
            left_col = columns[i]
            right_col = columns[j]
            score, pairs, row_count, covered_rows = compute_approx_fd(
                left_col, right_col, min_rows=min_rows
            )
            if score >= theta and pairs:
                candidates.append({
                    "pairs": [list(p) for p in pairs],
                    "theta": score,
                    "row_count": row_count,
                    "covered_rows": covered_rows,
                    "source_table_index": table_idx,
                    "left_column_index": i,
                    "right_column_index": j,
                    "source_metadata": metadata,
                })
    return candidates


# ---------------------------------------------------------------------------
# Parallel WP3 initial scoring (see specs/2026-06-03-parallel-wp3-...)
# ---------------------------------------------------------------------------

# Module-level globals populated in each worker process by
# `_init_scoring_worker`. They stay None in the parent.
_CANDIDATES = None
_USE_APPROX = None


def _init_scoring_worker(candidates, use_approx):
    """Pool initializer: stash candidates and use_approx in worker globals
    so per-task args can be just `(ci, cj)`."""
    global _CANDIDATES, _USE_APPROX
    _CANDIDATES = candidates
    _USE_APPROX = use_approx


def _score_edge_worker(edge):
    """Score one overlap edge. Reads from module globals set by
    `_init_scoring_worker`. Returns `(ci, cj, pos, neg)`."""
    ci, cj = edge
    bp = list(_CANDIDATES[ci]["pairs"])
    bq = list(_CANDIDATES[cj]["pairs"])
    ps = positive_score(bp, bq, use_approx=_USE_APPROX)
    ns = negative_score(bp, bq, use_approx=_USE_APPROX)
    return ci, cj, ps, ns


# ---------------------------------------------------------------------------
# Parallel PMI scoring
# ---------------------------------------------------------------------------

def parallel_score_corpus(
    corpus: list,
    index: tuple,
    n_workers: Optional[int] = None,
    chunk_size: int = 50,
) -> list:
    """Score all columns in corpus using multiple CPU cores.
    
    Args:
        corpus: list of (metadata, columns) tuples
        index: co-occurrence index tuple (cooc, value_count, N)
        n_workers: number of worker processes (default: CPU count)
        chunk_size: tables per chunk sent to each worker
        
    Returns:
        List of (table_idx, col_idx, values, score) tuples
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Prepare args for each table
    args = [(i, table, index) for i, table in enumerate(corpus)]

    results = []
    with mp.Pool(processes=n_workers) as pool:
        for scored_cols in pool.imap(
            _score_table_worker,
            args,
            chunksize=chunk_size
        ):
            results.extend(scored_cols)

    return results


# ---------------------------------------------------------------------------
# Parallel FD filtering
# ---------------------------------------------------------------------------

def parallel_fd_filter(
    filtered_corpus: list,
    theta: float = 0.95,
    min_rows: int = 3,
    n_workers: Optional[int] = None,
    chunk_size: int = 50,
) -> list:
    """Run FD filtering on corpus using multiple CPU cores.
    
    Args:
        filtered_corpus: list of filtered table dicts
        theta: approximate FD threshold
        min_rows: minimum rows for FD evaluation
        n_workers: number of worker processes
        chunk_size: tables per chunk
        
    Returns:
        List of candidate dicts
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [(i, table, theta, min_rows) for i, table in enumerate(filtered_corpus)]

    results = []
    with mp.Pool(processes=n_workers) as pool:
        for candidates in pool.imap(
            _fd_filter_worker,
            args,
            chunksize=chunk_size
        ):
            results.extend(candidates)

    return results


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark(
    corpus: list,
    index: tuple,
    n_workers_list: Optional[List[int]] = None,
) -> dict:
    """Benchmark parallel vs sequential PMI scoring.
    
    Args:
        corpus: list of (metadata, columns) tuples
        index: co-occurrence index
        n_workers_list: list of worker counts to test
        
    Returns:
        Dict of {n_workers: time_seconds}
    """
    from npmi import score_corpus as sequential_score

    if n_workers_list is None:
        max_workers = mp.cpu_count()
        n_workers_list = [1, 2, max_workers]

    results = {}

    # Sequential baseline
    print(f"  Benchmarking PMI scoring on {len(corpus)} tables...")
    print(f"  Available CPU cores: {mp.cpu_count()}")
    print()

    t0 = time.time()
    sequential_score(corpus, index)
    seq_time = time.time() - t0
    results["sequential"] = seq_time
    print(f"  Sequential (1 core):  {seq_time:.2f}s")

    # Parallel with different worker counts
    for n in n_workers_list:
        if n <= 1:
            continue
        t0 = time.time()
        parallel_score_corpus(corpus, index, n_workers=n)
        par_time = time.time() - t0
        results[f"parallel_{n}"] = par_time
        speedup = seq_time / par_time if par_time > 0 else 1.0
        print(f"  Parallel ({n} cores):  {par_time:.2f}s  (speedup: {speedup:.1f}x)")

    print()
    print(f"  Note: Parallelization overhead dominates on small corpora.")
    print(f"  Speedup expected on corpora with 10,000+ tables.")
    print(f"  Recommended minimum corpus size: ~10,000 tables")
    
    return results

def should_parallelize(corpus_size: int, threshold: int = 10000) -> bool:
    """Decide if parallelization is worth it based on corpus size.
    
    For small corpora, multiprocessing overhead exceeds the benefit.
    Rule of thumb: only parallelize for corpora with 10,000+ tables.
    
    Args:
        corpus_size: number of tables in corpus
        threshold: minimum corpus size for parallelization
        
    Returns:
        True if parallelization is recommended
    """
    return corpus_size >= threshold


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo(corpus_path: str = "dev_chunks/chunk_1_mini_1000.json") -> None:
    """Run benchmark on real data."""
    import json
    from data_loader import load_corpus
    from cooccurrence_index import build_cooccurrence_index

    print("Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} tables")

    print("Building index...")
    index = build_cooccurrence_index(corpus)

    print("\nRunning benchmark...")
    benchmark(corpus, index)


if __name__ == "__main__":
    demo()