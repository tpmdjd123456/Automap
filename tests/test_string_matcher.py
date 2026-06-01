"""Tests for string_matcher.py"""

import pytest
from string_matcher import (
    normalize,
    tokenize,
    exact_match,
    edit_distance_match,
    token_set_match,
    abbreviation_match,
    EnhancedStringMatcher,
)


# ---------------------------------------------------------------------------
# Tests for normalize
# ---------------------------------------------------------------------------

def test_normalize_lowercase():
    assert normalize("FRANCE") == "france"

def test_normalize_punctuation():
    assert normalize("U.S.A.") == "usa"

def test_normalize_parentheses():
    assert normalize("American Samoa (US)") == "american samoa"

def test_normalize_whitespace():
    assert normalize("New  York   City") == "new york city"

def test_normalize_dashes():
    assert normalize("hip-hop") == "hip hop"


# ---------------------------------------------------------------------------
# Tests for exact_match
# ---------------------------------------------------------------------------

def test_exact_match_same():
    assert exact_match("France", "France") is True

def test_exact_match_case():
    assert exact_match("FRANCE", "france") is True

def test_exact_match_punctuation():
    assert exact_match("U.S.A.", "USA") is True

def test_exact_match_parentheses():
    assert exact_match("American Samoa (US)", "American Samoa") is True

def test_exact_match_false():
    assert exact_match("France", "Germany") is False


# ---------------------------------------------------------------------------
# Tests for edit_distance_match
# ---------------------------------------------------------------------------

def test_edit_distance_close():
    assert edit_distance_match("Korea Republic of", "Korea, Republic of") is True

def test_edit_distance_too_far():
    assert edit_distance_match("France", "Germany") is False

def test_edit_distance_exact():
    assert edit_distance_match("France", "France") is True


# ---------------------------------------------------------------------------
# Tests for token_set_match
# ---------------------------------------------------------------------------

def test_token_set_reorder():
    assert token_set_match("Republic of Korea", "Korea Republic") is True

def test_token_set_reorder_long():
    assert token_set_match("United States Virgin Islands", "Virgin Islands United States") is True

def test_token_set_different():
    assert token_set_match("France", "Germany") is False

def test_token_set_empty():
    assert token_set_match("", "France") is False


# ---------------------------------------------------------------------------
# Tests for abbreviation_match
# ---------------------------------------------------------------------------

def test_abbreviation_nyc():
    assert abbreviation_match("NYC", "New York City") is True

def test_abbreviation_usa():
    assert abbreviation_match("USA", "United States of America") is True

def test_abbreviation_uk():
    assert abbreviation_match("UK", "United Kingdom") is True

def test_abbreviation_reverse():
    assert abbreviation_match("New York City", "NYC") is True

def test_abbreviation_false():
    assert abbreviation_match("France", "Germany") is False

def test_abbreviation_too_short():
    assert abbreviation_match("A", "Apple") is False


# ---------------------------------------------------------------------------
# Tests for EnhancedStringMatcher
# ---------------------------------------------------------------------------

def test_enhanced_exact():
    m = EnhancedStringMatcher()
    matched, strategy = m.are_match("France", "France")
    assert matched is True
    assert strategy == "exact"

def test_enhanced_token_set():
    m = EnhancedStringMatcher()
    matched, strategy = m.are_match("Republic of Korea", "Korea Republic")
    assert matched is True
    assert strategy == "token_set"

def test_enhanced_abbreviation():
    m = EnhancedStringMatcher()
    matched, strategy = m.are_match("NYC", "New York City")
    assert matched is True
    assert strategy == "abbreviation"

def test_enhanced_no_match():
    m = EnhancedStringMatcher()
    matched, strategy = m.are_match("France", "Germany")
    assert matched is False
    assert strategy == "no_match"

def test_enhanced_empty():
    m = EnhancedStringMatcher()
    matched, strategy = m.are_match("", "France")
    assert matched is False

def test_similarity_score_exact():
    m = EnhancedStringMatcher()
    assert m.similarity_score("France", "France") == 1.0

def test_similarity_score_token():
    m = EnhancedStringMatcher()
    assert m.similarity_score("Republic of Korea", "Korea Republic") == 0.7

def test_similarity_score_abbreviation():
    m = EnhancedStringMatcher()
    assert m.similarity_score("NYC", "New York City") == 0.6

def test_similarity_score_no_match():
    m = EnhancedStringMatcher()
    assert m.similarity_score("France", "Germany") == 0.0

def test_enhanced_improvements_over_original():
    """Enhanced matcher should find more matches than original."""
    m = EnhancedStringMatcher()
    pairs = [
        ("Republic of Korea", "Korea Republic"),
        ("NYC", "New York City"),
        ("USA", "United States of America"),
    ]
    for a, b in pairs:
        assert edit_distance_match(a, b) is False
        matched, _ = m.are_match(a, b)
        assert matched is True