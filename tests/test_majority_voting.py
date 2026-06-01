"""Tests for majority_voting.py"""

import pytest
from majority_voting import (
    count_pair_support,
    majority_vote,
)


# ---------------------------------------------------------------------------
# Tests for count_pair_support
# ---------------------------------------------------------------------------

def test_count_pair_support_basic():
    """Pairs appearing in multiple candidates get higher counts."""
    candidates = [
        {"pairs": [("algeria", "alg"), ("albania", "alb")]},
        {"pairs": [("algeria", "alg"), ("angola", "ago")]},
        {"pairs": [("algeria", "dza")]},
    ]
    support = count_pair_support([0, 1, 2], candidates)
    assert support[("algeria", "alg")] == 2
    assert support[("algeria", "dza")] == 1
    assert support[("albania", "alb")] == 1


def test_count_pair_support_empty():
    """Empty candidate indices returns empty support."""
    candidates = [{"pairs": [("algeria", "alg")]}]
    support = count_pair_support([], candidates)
    assert len(support) == 0


def test_count_pair_support_single():
    """Single candidate returns count of 1 for each pair."""
    candidates = [{"pairs": [("algeria", "alg"), ("albania", "alb")]}]
    support = count_pair_support([0], candidates)
    assert support[("algeria", "alg")] == 1
    assert support[("albania", "alb")] == 1


# ---------------------------------------------------------------------------
# Tests for majority_vote
# ---------------------------------------------------------------------------

def test_majority_vote_no_conflicts():
    """Clean pairs pass through unchanged."""
    pairs = [("algeria", "alg"), ("albania", "alb"), ("angola", "ago")]
    support = {("algeria", "alg"): 3, ("albania", "alb"): 2, ("angola", "ago"): 2}
    result = majority_vote(pairs, support)
    assert not any(
        sum(1 for l, r in result if l == left) > 1
        for left, _ in result
    )


def test_majority_vote_simple_conflict():
    """Majority right value wins."""
    pairs = [("algeria", "alg"), ("algeria", "dza"), ("albania", "alb")]
    support = {("algeria", "alg"): 5, ("algeria", "dza"): 1, ("albania", "alb"): 3}
    result = majority_vote(pairs, support)
    algeria_pairs = [(l, r) for l, r in result if l == "algeria"]
    assert len(algeria_pairs) == 1
    assert algeria_pairs[0][1] == "alg"


def test_majority_vote_keeps_winner():
    """The value with highest support is always kept."""
    pairs = [
        ("chiip", "no"),
        ("chiip", "yes"),
        ("argon", "yes"),
    ]
    support = {("chiip", "yes"): 8, ("chiip", "no"): 2, ("argon", "yes"): 5}
    result = majority_vote(pairs, support)
    chiip_pairs = [(l, r) for l, r in result if l == "chiip"]
    assert len(chiip_pairs) == 1
    assert chiip_pairs[0][1] == "yes"


def test_majority_vote_no_conflicts_remain():
    """After majority voting no left value maps to two right values."""
    pairs = [
        ("algeria", "alg"), ("algeria", "dza"),
        ("albania", "alb"), ("albania", "alb2"),
        ("angola", "ago"),
    ]
    support = {
        ("algeria", "alg"): 3, ("algeria", "dza"): 1,
        ("albania", "alb"): 4, ("albania", "alb2"): 1,
        ("angola", "ago"): 2,
    }
    result = majority_vote(pairs, support)
    left_values = [l for l, r in result]
    assert len(left_values) == len(set(left_values))


def test_majority_vote_empty():
    """Empty pairs returns empty."""
    assert majority_vote([], {}) == []


def test_majority_vote_result_is_subset():
    """Result must always be a subset of input left values."""
    pairs = [("algeria", "alg"), ("algeria", "dza"), ("albania", "alb")]
    support = {("algeria", "alg"): 3, ("algeria", "dza"): 1, ("albania", "alb"): 2}
    result = majority_vote(pairs, support)
    input_lefts = set(l for l, r in pairs)
    result_lefts = set(l for l, r in result)
    assert result_lefts.issubset(input_lefts)