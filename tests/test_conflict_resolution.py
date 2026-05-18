"""Tests for conflict_resolution.py (WP4)."""

import pytest
from conflict_resolution import (
    find_conflicts,
    has_conflicts,
    resolve_conflicts,
)


# ---------------------------------------------------------------------------
# Tests for find_conflicts
# ---------------------------------------------------------------------------

def test_find_conflicts_no_conflicts():
    """Clean mapping should return no conflicts."""
    pairs = [("Algeria", "ALG"), ("Albania", "ALB"), ("Afghanistan", "AFG")]
    assert find_conflicts(pairs) == {}


def test_find_conflicts_one_conflict():
    """One left value with two different right values."""
    pairs = [("Algeria", "ALG"), ("Algeria", "DZA"), ("Albania", "ALB")]
    result = find_conflicts(pairs)
    assert "Algeria" in result
    assert sorted(result["Algeria"]) == ["ALG", "DZA"]


def test_find_conflicts_multiple_conflicts():
    """Multiple conflicting left values."""
    pairs = [
        ("Algeria", "ALG"),
        ("Algeria", "DZA"),
        ("Albania", "ALB"),
        ("Albania", "ALB2"),
    ]
    result = find_conflicts(pairs)
    assert "Algeria" in result
    assert "Albania" in result


def test_find_conflicts_empty():
    """Empty input should return no conflicts."""
    assert find_conflicts([]) == {}


def test_find_conflicts_single_pair():
    """Single pair can never conflict with itself."""
    assert find_conflicts([("Algeria", "ALG")]) == {}


# ---------------------------------------------------------------------------
# Tests for has_conflicts
# ---------------------------------------------------------------------------

def test_has_conflicts_true():
    pairs = [("Algeria", "ALG"), ("Algeria", "DZA")]
    assert has_conflicts(pairs) is True


def test_has_conflicts_false():
    pairs = [("Algeria", "ALG"), ("Albania", "ALB")]
    assert has_conflicts(pairs) is False


def test_has_conflicts_empty():
    assert has_conflicts([]) is False


# ---------------------------------------------------------------------------
# Tests for resolve_conflicts
# ---------------------------------------------------------------------------

def test_resolve_conflicts_no_conflicts():
    """Clean input should pass through unchanged."""
    pairs = [("Algeria", "ALG"), ("Albania", "ALB"), ("Afghanistan", "AFG")]
    result = resolve_conflicts(pairs)
    assert sorted(result) == sorted(pairs)


def test_resolve_conflicts_simple():
    """One conflict: should remove one of the two conflicting pairs."""
    pairs = [("Algeria", "ALG"), ("Algeria", "DZA"), ("Albania", "ALB")]
    result = resolve_conflicts(pairs)
    assert not has_conflicts(result)
    # Albania should survive
    assert ("Albania", "ALB") in result
    # Only one Algeria pair should remain
    algeria_pairs = [p for p in result if p[0] == "Algeria"]
    assert len(algeria_pairs) == 1


def test_resolve_conflicts_empty():
    """Empty input should return empty."""
    assert resolve_conflicts([]) == []


def test_resolve_conflicts_all_conflict():
    """All pairs conflict: should end up with one pair per left value."""
    pairs = [
        ("A", "1"), ("A", "2"), ("A", "3"),
    ]
    result = resolve_conflicts(pairs)
    assert not has_conflicts(result)
    assert len(result) == 1


def test_resolve_conflicts_result_is_subset():
    """Result must always be a subset of the input."""
    pairs = [
        ("Algeria", "ALG"), ("Algeria", "DZA"),
        ("Albania", "ALB"), ("Albania", "ALB2"),
        ("Afghanistan", "AFG"),
    ]
    result = resolve_conflicts(pairs)
    for pair in result:
        assert pair in pairs


def test_resolve_conflicts_no_conflicts_remain():
    """After resolution there must be zero conflicts."""
    pairs = [
        ("Algeria", "ALG"), ("Algeria", "DZA"),
        ("Albania", "ALB"), ("Albania", "ALB2"),
        ("Congo", "COD"), ("Congo", "COG"),
    ]
    result = resolve_conflicts(pairs)
    assert not has_conflicts(result)