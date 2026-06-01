"""Tests for confidence_scorer.py"""

import pytest
from confidence_scorer import ConfidenceScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_candidates():
    return [
        {"pairs": [("france", "fra"), ("germany", "deu")], "theta": 1.0},
        {"pairs": [("france", "fra"), ("japan", "jpn")], "theta": 0.98},
        {"pairs": [("france", "deu")], "theta": 0.95},  # conflict!
    ]


@pytest.fixture
def simple_synthesized():
    return {
        "partition_id": 0,
        "candidate_indices": [0, 1, 2],
        "pairs": [["france", "fra"], ["germany", "deu"], ["japan", "jpn"], ["france", "deu"]],
        "size": 4,
        "num_source_tables": 3,
    }


@pytest.fixture
def scorer():
    return ConfidenceScorer()


# ---------------------------------------------------------------------------
# Tests for score_partition
# ---------------------------------------------------------------------------

def test_score_partition_returns_all_pairs(scorer, simple_synthesized, simple_candidates):
    """All pairs in synthesized mapping should be scored."""
    results = scorer.score_partition(simple_synthesized, simple_candidates)
    assert len(results) == 4


def test_score_partition_confidence_range(scorer, simple_synthesized, simple_candidates):
    """All confidence scores should be between 0 and 1."""
    results = scorer.score_partition(simple_synthesized, simple_candidates)
    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0


def test_score_partition_sorted_descending(scorer, simple_synthesized, simple_candidates):
    """Results should be sorted by confidence descending."""
    results = scorer.score_partition(simple_synthesized, simple_candidates)
    confs = [r["confidence"] for r in results]
    assert confs == sorted(confs, reverse=True)


def test_score_partition_high_support_higher_confidence(scorer, simple_synthesized, simple_candidates):
    """Pairs with higher support should have higher confidence."""
    results = scorer.score_partition(simple_synthesized, simple_candidates)
    pair_confs = {tuple(r["pair"]): r["confidence"] for r in results}
    # france→fra appears in 2 candidates, france→deu in only 1
    assert pair_confs[("france", "fra")] > pair_confs[("france", "deu")]


def test_score_partition_with_resolved(scorer, simple_synthesized, simple_candidates):
    """Pairs that survived conflict resolution should have higher confidence."""
    resolved = [("france", "fra"), ("germany", "deu"), ("japan", "jpn")]
    results_with = scorer.score_partition(simple_synthesized, simple_candidates, resolved)
    results_without = scorer.score_partition(simple_synthesized, simple_candidates)
    # france→fra should have higher confidence when it survived resolution
    with_confs = {tuple(r["pair"]): r["confidence"] for r in results_with}
    without_confs = {tuple(r["pair"]): r["confidence"] for r in results_without}
    assert with_confs[("france", "fra")] >= without_confs[("france", "fra")]


def test_score_partition_survived_conflict_field(scorer, simple_synthesized, simple_candidates):
    """survived_conflict field should be set correctly."""
    resolved = [("france", "fra"), ("germany", "deu")]
    results = scorer.score_partition(simple_synthesized, simple_candidates, resolved)
    pair_survived = {tuple(r["pair"]): r["survived_conflict"] for r in results}
    assert pair_survived[("france", "fra")] is True
    assert pair_survived[("france", "deu")] is False


def test_score_partition_support_count(scorer, simple_synthesized, simple_candidates):
    """Support count should reflect number of candidates containing pair."""
    results = scorer.score_partition(simple_synthesized, simple_candidates)
    pair_support = {tuple(r["pair"]): r["support_count"] for r in results}
    assert pair_support[("france", "fra")] == 2
    assert pair_support[("france", "deu")] == 1


# ---------------------------------------------------------------------------
# Tests for ConfidenceScorer weights
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    """Default weights should sum to 1.0."""
    scorer = ConfidenceScorer()
    total = scorer.weight_support + scorer.weight_theta + scorer.weight_conflict
    assert abs(total - 1.0) < 1e-6


def test_invalid_weights():
    """Weights not summing to 1.0 should raise assertion error."""
    with pytest.raises(AssertionError):
        ConfidenceScorer(weight_support=0.5, weight_theta=0.5, weight_conflict=0.5)


def test_custom_weights(simple_synthesized, simple_candidates):
    """Custom weights should produce different scores."""
    scorer1 = ConfidenceScorer(weight_support=0.8, weight_theta=0.1, weight_conflict=0.1)
    scorer2 = ConfidenceScorer(weight_support=0.1, weight_theta=0.1, weight_conflict=0.8)
    results1 = scorer1.score_partition(simple_synthesized, simple_candidates)
    results2 = scorer2.score_partition(simple_synthesized, simple_candidates)
    confs1 = {tuple(r["pair"]): r["confidence"] for r in results1}
    confs2 = {tuple(r["pair"]): r["confidence"] for r in results2}
    # Results should differ with different weights
    assert confs1 != confs2