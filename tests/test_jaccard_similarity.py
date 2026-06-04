"""Tests for jaccard_similarity.py"""

import pytest
from jaccard_similarity import (
    char_ngrams,
    word_ngrams,
    jaccard,
    char_jaccard,
    word_jaccard,
    JaccardMatcher,
)


# ---------------------------------------------------------------------------
# Tests for char_ngrams
# ---------------------------------------------------------------------------

def test_char_ngrams_basic():
    result = char_ngrams("france", 2)
    assert "fr" in result
    assert "ra" in result
    assert "an" in result

def test_char_ngrams_empty():
    assert char_ngrams("", 2) == set()

def test_char_ngrams_short():
    result = char_ngrams("ab", 2)
    assert "ab" in result

def test_char_ngrams_trigram():
    result = char_ngrams("france", 3)
    assert "fra" in result
    assert "ran" in result


# ---------------------------------------------------------------------------
# Tests for word_ngrams
# ---------------------------------------------------------------------------

def test_word_ngrams_basic():
    result = word_ngrams("new york city")
    assert "new" in result
    assert "york" in result
    assert "city" in result

def test_word_ngrams_empty():
    assert word_ngrams("") == set()

def test_word_ngrams_single():
    result = word_ngrams("france")
    assert "france" in result


# ---------------------------------------------------------------------------
# Tests for jaccard
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    s = {"a", "b", "c"}
    assert jaccard(s, s) == 1.0

def test_jaccard_empty_both():
    assert jaccard(set(), set()) == 1.0

def test_jaccard_one_empty():
    assert jaccard({"a"}, set()) == 0.0

def test_jaccard_no_overlap():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0

def test_jaccard_partial():
    a = {"a", "b", "c"}
    b = {"b", "c", "d"}
    assert jaccard(a, b) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests for char_jaccard and word_jaccard
# ---------------------------------------------------------------------------

def test_char_jaccard_identical():
    assert char_jaccard("france", "france") == 1.0

def test_char_jaccard_different():
    assert char_jaccard("france", "germany") < 0.3

def test_word_jaccard_reorder():
    """Word Jaccard should handle word reordering."""
    score = word_jaccard("Republic of Korea", "Korea Republic")
    assert score > 0.5

def test_word_jaccard_city():
    score = word_jaccard("City of New York", "New York City")
    assert score > 0.5


# ---------------------------------------------------------------------------
# Tests for JaccardMatcher
# ---------------------------------------------------------------------------

def test_matcher_identical():
    m = JaccardMatcher()
    matched, score = m.are_match("France", "France")
    assert matched is True
    assert score == 1.0

def test_matcher_reorder():
    m = JaccardMatcher()
    matched, score = m.are_match("Republic of Korea", "Korea Republic")
    assert matched is True

def test_matcher_city_reorder():
    m = JaccardMatcher()
    matched, score = m.are_match("City of New York", "New York City")
    assert matched is True

def test_matcher_different():
    m = JaccardMatcher()
    matched, score = m.are_match("France", "Germany")
    assert matched is False

def test_matcher_empty():
    m = JaccardMatcher()
    matched, score = m.are_match("", "France")
    assert matched is False

def test_matcher_better_than_jaro_winkler_reorder():
    """Jaccard should handle word reordering better than Jaro-Winkler."""
    from jaro_winkler import JaroWinklerMatcher
    jw = JaroWinklerMatcher()
    jac = JaccardMatcher()
    pairs = [
        ("Republic of Korea", "Korea Republic"),
        ("City of New York", "New York City"),
        ("United States of America", "America United States"),
    ]
    for a, b in pairs:
        jw_match, _ = jw.are_match(a, b)
        jac_match, _ = jac.are_match(a, b)
        assert jac_match is True
        assert jw_match is False

def test_matcher_custom_threshold():
    m_strict = JaccardMatcher(threshold=0.9)
    m_loose  = JaccardMatcher(threshold=0.3)
    a, b = "Republic of Korea", "Korea Republic"
    strict_match, _ = m_strict.are_match(a, b)
    loose_match, _  = m_loose.are_match(a, b)
    assert loose_match is True