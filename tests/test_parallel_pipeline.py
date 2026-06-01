"""Tests for parallel_pipeline.py"""

import pytest
from parallel_pipeline import (
    parallel_score_corpus,
    parallel_fd_filter,
    should_parallelize,
)
from npmi import score_corpus as sequential_score
from cooccurrence_index import build_cooccurrence_index
from fd_filter import filter_candidates_by_fd
from filter import filter_corpus, rebuild_filtered_corpus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(synthetic_corpus):
    return synthetic_corpus


@pytest.fixture
def index(synthetic_corpus):
    return build_cooccurrence_index(synthetic_corpus)


@pytest.fixture
def filtered(synthetic_corpus, index):
    scored = sequential_score(synthetic_corpus, index)
    kept, _ = filter_corpus(scored, threshold=0.3)
    return rebuild_filtered_corpus(synthetic_corpus, kept)


# ---------------------------------------------------------------------------
# Tests for should_parallelize
# ---------------------------------------------------------------------------

def test_should_parallelize_small():
    """Small corpus should not parallelize."""
    assert should_parallelize(100) is False
    assert should_parallelize(1000) is False
    assert should_parallelize(9999) is False


def test_should_parallelize_large():
    """Large corpus should parallelize."""
    assert should_parallelize(10000) is True
    assert should_parallelize(100000) is True


def test_should_parallelize_custom_threshold():
    """Custom threshold should work."""
    assert should_parallelize(500, threshold=1000) is False
    assert should_parallelize(1000, threshold=1000) is True


# ---------------------------------------------------------------------------
# Tests for parallel_score_corpus
# ---------------------------------------------------------------------------

def test_parallel_score_same_results(corpus, index):
    """Parallel scoring should produce same results as sequential."""
    sequential = sequential_score(corpus, index)
    parallel = parallel_score_corpus(corpus, index, n_workers=2)

    assert len(parallel) == len(sequential)

    seq_dict = {(t, c): s for t, c, _, s in sequential}
    par_dict = {(t, c): s for t, c, _, s in parallel}

    for key in seq_dict:
        assert abs(seq_dict[key] - par_dict[key]) < 1e-6


def test_parallel_score_single_worker(corpus, index):
    """Single worker parallel should match sequential."""
    sequential = sequential_score(corpus, index)
    parallel = parallel_score_corpus(corpus, index, n_workers=1)
    assert len(parallel) == len(sequential)


def test_parallel_score_empty_corpus(index):
    """Empty corpus should return empty results."""
    result = parallel_score_corpus([], index, n_workers=2)
    assert result == []


# ---------------------------------------------------------------------------
# Tests for parallel_fd_filter
# ---------------------------------------------------------------------------

def test_parallel_fd_same_results(corpus, index):
    """Parallel FD filter should produce same candidates as sequential."""
    from filter import filter_corpus, rebuild_filtered_corpus
    from npmi import score_corpus
    
    scored = score_corpus(corpus, index)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered_corpus = rebuild_filtered_corpus(corpus, kept)
    
    # Save to temp JSONL and reload as dicts for filter_candidates_by_fd
    import json, tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        from filter import save_filtered_corpus
        tmp_path = f.name
    
    save_filtered_corpus(filtered_corpus, tmp_path)
    
    filtered_dicts = []
    with open(tmp_path) as f:
        for line in f:
            line = line.strip()
            if line:
                filtered_dicts.append(json.loads(line))
    os.unlink(tmp_path)
    
    sequential = list(filter_candidates_by_fd(filtered_dicts))
    parallel = parallel_fd_filter(filtered_dicts, n_workers=2)
    assert len(parallel) == len(sequential)


def test_parallel_fd_empty():
    """Empty corpus should return empty candidates."""
    result = parallel_fd_filter([], n_workers=2)
    assert result == []