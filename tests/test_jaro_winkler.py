"""Tests for jaro_winkler.py"""

import pytest
from jaro_winkler import jaro, jaro_winkler, JaroWinklerMatcher, normalize


# ---------------------------------------------------------------------------
# Tests for jaro
# ---------------------------------------------------------------------------

def test_jaro_identical():
    assert jaro("france", "france") == 1.0

def test_jaro_empty():
    assert jaro("", "france") == 0.0
    assert jaro("france", "") == 0.0

def test_jaro_completely_different():
    assert jaro("france", "germany") < 0.75

def test_jaro_similar():
    assert jaro("smith", "smyth") > 0.8

def test_jaro_range():
    score = jaro("korea", "republic")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests for jaro_winkler
# ---------------------------------------------------------------------------

def test_jaro_winkler_identical():
    assert jaro_winkler("france", "france") == 1.0

def test_jaro_winkler_higher_than_jaro_with_prefix():
    """Jaro-Winkler should score higher than Jaro when prefix matches."""
    j = jaro("johnathan", "jonathan")
    jw = jaro_winkler("johnathan", "jonathan")
    assert jw >= j

def test_jaro_winkler_range():
    score = jaro_winkler("france", "germany")
    assert 0.0 <= score <= 1.0

def test_jaro_winkler_similar_names():
    assert jaro_winkler("smith", "smyth") > 0.85

def test_jaro_winkler_colour_color():
    assert jaro_winkler("colour", "color") > 0.9


# ---------------------------------------------------------------------------
# Tests for JaroWinklerMatcher
# ---------------------------------------------------------------------------

def test_matcher_identical():
    m = JaroWinklerMatcher()
    matched, score = m.are_match("France", "France")
    assert matched is True
    assert score == 1.0

def test_matcher_microsoft():
    m = JaroWinklerMatcher()
    matched, score = m.are_match("Microsoft Corp", "Microsoft Corporation")
    assert matched is True
    assert score > 0.85

def test_matcher_new_york():
    m = JaroWinklerMatcher()
    matched, score = m.are_match("New York", "New York City")
    assert matched is True

def test_matcher_france_germany():
    m = JaroWinklerMatcher()
    matched, score = m.are_match("France", "Germany")
    assert matched is False

def test_matcher_empty():
    m = JaroWinklerMatcher()
    matched, score = m.are_match("", "France")
    assert matched is False

def test_matcher_custom_threshold():
    """Lower threshold should match more pairs."""
    m_strict = JaroWinklerMatcher(threshold=0.95)
    m_loose  = JaroWinklerMatcher(threshold=0.7)
    a, b = "colour", "color"
    strict_match, _ = m_strict.are_match(a, b)
    loose_match, _  = m_loose.are_match(a, b)
    assert loose_match is True

def test_matcher_better_than_edit_distance():
    """Jaro-Winkler should find matches that edit distance misses."""
    from string_matcher import edit_distance_match
    m = JaroWinklerMatcher()
    pairs = [
        ("Microsoft Corp", "Microsoft Corporation"),
        ("New York", "New York City"),
        ("centre", "center"),
    ]
    for a, b in pairs:
        ed = edit_distance_match(a, b)
        jw, _ = m.are_match(a, b)
        assert jw is True
        assert ed is False