"""Tests for synonym_detector.py"""

import pytest
import os
import tempfile
from synonym_detector import SynonymDetector, boost_compatibility_with_synonyms


# ---------------------------------------------------------------------------
# Tests for SynonymDetector
# ---------------------------------------------------------------------------

def test_builtin_synonyms_loaded():
    """Built-in synonyms should be loaded by default."""
    sd = SynonymDetector()
    assert sd.are_synonyms("USA", "United States")
    assert sd.are_synonyms("UK", "United Kingdom")
    assert sd.are_synonyms("CA", "California")


def test_no_builtins():
    """Without builtins, no synonyms should be loaded."""
    sd = SynonymDetector(use_builtins=False)
    assert not sd.are_synonyms("USA", "United States")


def test_add_synonym_basic():
    """Adding a synonym pair should make them synonyms."""
    sd = SynonymDetector(use_builtins=False)
    sd.add_synonym("hello", "hi")
    assert sd.are_synonyms("hello", "hi")
    assert sd.are_synonyms("hi", "hello")


def test_add_synonym_transitive():
    """Synonyms should be transitive: if A=B and B=C then A=C."""
    sd = SynonymDetector(use_builtins=False)
    sd.add_synonym("a", "b")
    sd.add_synonym("b", "c")
    assert sd.are_synonyms("a", "c")
    assert sd.are_synonyms("c", "a")


def test_are_synonyms_same_value():
    """A value is always a synonym of itself."""
    sd = SynonymDetector()
    assert sd.are_synonyms("france", "france")
    assert sd.are_synonyms("USA", "USA")


def test_are_synonyms_false():
    """Non-synonyms should return False."""
    sd = SynonymDetector()
    assert not sd.are_synonyms("France", "Germany")
    assert not sd.are_synonyms("USA", "France")


def test_are_synonyms_case_insensitive():
    """Synonym checks should be case insensitive."""
    sd = SynonymDetector()
    assert sd.are_synonyms("usa", "UNITED STATES")
    assert sd.are_synonyms("USA", "united states")


def test_get_synonyms():
    """get_synonyms should return all synonyms for a value."""
    sd = SynonymDetector(use_builtins=False)
    sd.add_synonym("usa", "united states")
    sd.add_synonym("usa", "united states of america")
    synonyms = sd.get_synonyms("usa")
    assert "usa" in synonyms
    assert "united states" in synonyms
    assert "united states of america" in synonyms


def test_get_synonyms_unknown():
    """get_synonyms for unknown value returns just that value."""
    sd = SynonymDetector(use_builtins=False)
    synonyms = sd.get_synonyms("unknown_value")
    assert "unknown_value" in synonyms


def test_load_from_csv(tmp_path):
    """Loading synonyms from CSV should work correctly."""
    csv_file = tmp_path / "synonyms.csv"
    csv_file.write_text("hello,hi\ngoodbye,bye\ntest,exam\n")
    sd = SynonymDetector(use_builtins=False)
    count = sd.load_from_csv(str(csv_file))
    assert count == 3
    assert sd.are_synonyms("hello", "hi")
    assert sd.are_synonyms("goodbye", "bye")
    assert sd.are_synonyms("test", "exam")


def test_load_from_csv_missing_file():
    """Loading from missing file should not crash."""
    sd = SynonymDetector(use_builtins=False)
    count = sd.load_from_csv("nonexistent.csv")
    assert count == 0


def test_save_and_load_csv(tmp_path):
    """Save and reload synonyms should produce same results."""
    sd = SynonymDetector(use_builtins=False)
    sd.add_synonym("hello", "hi")
    sd.add_synonym("goodbye", "bye")
    path = str(tmp_path / "saved.csv")
    sd.save_to_csv(path)
    sd2 = SynonymDetector(use_builtins=False)
    sd2.load_from_csv(path)
    assert sd2.are_synonyms("hello", "hi")
    assert sd2.are_synonyms("goodbye", "bye")


# ---------------------------------------------------------------------------
# Tests for boost_compatibility_with_synonyms
# ---------------------------------------------------------------------------

def test_boost_no_synonyms():
    """No synonym boost when detector has no synonyms."""
    sd = SynonymDetector(use_builtins=False)
    pairs_a = [("usa", "north america")]
    pairs_b = [("united states", "north america")]
    boost = boost_compatibility_with_synonyms(pairs_a, pairs_b, sd)
    assert boost == 0


def test_boost_with_synonyms():
    """Synonym boost should count synonym matches."""
    sd = SynonymDetector(use_builtins=False)
    sd.add_synonym("usa", "united states")
    sd.add_synonym("north america", "americas")
    pairs_a = [("usa", "north america")]
    pairs_b = [("united states", "americas")]
    boost = boost_compatibility_with_synonyms(pairs_a, pairs_b, sd)
    assert boost == 1


def test_boost_exact_matches_not_counted():
    """Exact matches should not be counted as synonym boosts."""
    sd = SynonymDetector(use_builtins=False)
    sd.add_synonym("usa", "united states")
    pairs_a = [("usa", "north america")]
    pairs_b = [("usa", "north america")]
    boost = boost_compatibility_with_synonyms(pairs_a, pairs_b, sd)
    assert boost == 0


def test_boost_empty_pairs():
    """Empty pairs should return 0 boost."""
    sd = SynonymDetector()
    assert boost_compatibility_with_synonyms([], [], sd) == 0
    assert boost_compatibility_with_synonyms([("a","b")], [], sd) == 0