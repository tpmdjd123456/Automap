"""Enhanced Approximate String Matching — improvement to WP3 §4.1.

The paper uses basic edit distance with a fractional threshold.
This module improves on that with:

1. Punctuation normalization — remove dots, commas etc. before comparing
2. Token-based matching — "Republic of Korea" = "Korea Republic"
3. Abbreviation detection — "NYC" matches "New York City"
4. Combined scoring — use all three signals together

Usage:
    from string_matcher import EnhancedStringMatcher
    matcher = EnhancedStringMatcher()
    matcher.are_match("Korea Republic", "Republic of Korea")  # True
    matcher.are_match("NYC", "New York City")                 # True
    matcher.are_match("U.S.A.", "USA")                        # True
"""

from __future__ import annotations

import re
import string
from typing import List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for comparison.
    
    - Lowercase
    - Remove content in parentheses
    - Remove punctuation (dots, commas, dashes, parentheses)
    - Collapse whitespace
    """
    text = text.lower().strip()
    # Remove parenthetical content e.g. "American Samoa (US)" -> "American Samoa"
    text = re.sub(r"\(.*?\)", "", text)
    # Remove dots from abbreviations e.g. "U.S.A." -> "USA"
    text = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", text)
    # Remove remaining punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    return normalize(text).split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common stopwords from token list."""
    stopwords = {"of", "the", "a", "an", "and", "or", "in", "at", "to", "for"}
    return [t for t in tokens if t not in stopwords]


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------

def exact_match(a: str, b: str) -> bool:
    """Check if two strings match after normalization."""
    return normalize(a) == normalize(b)


def edit_distance_match(a: str, b: str, fed: float = 0.2, ked: int = 10) -> bool:
    """Paper's original edit distance matching (§4.1 Algorithm 2).
    
    Uses fractional threshold: threshold = min(floor(|a|*fed), floor(|b|*fed), ked)
    """
    a_norm = normalize(a)
    b_norm = normalize(b)

    if a_norm == b_norm:
        return True

    # Ensure a is shorter
    if len(a_norm) > len(b_norm):
        a_norm, b_norm = b_norm, a_norm

    threshold = min(
        int(len(a_norm) * fed),
        int(len(b_norm) * fed),
        ked
    )

    if threshold == 0:
        return a_norm == b_norm

    # Band DP (Ukkonen-style) — only compute within threshold band
    dist = [[float('inf')] * (len(b_norm) + 1) for _ in range(len(a_norm) + 1)]
    for i in range(len(a_norm) + 1):
        dist[i][0] = i
    for j in range(len(b_norm) + 1):
        dist[0][j] = j

    for i in range(1, len(a_norm) + 1):
        lower = max(1, i - threshold)
        upper = min(len(b_norm), i + threshold)
        for j in range(lower, upper + 1):
            cost = 0 if a_norm[i-1] == b_norm[j-1] else 1
            dist[i][j] = min(
                dist[i-1][j] + 1,
                dist[i][j-1] + 1,
                dist[i-1][j-1] + cost
            )

    return dist[len(a_norm)][len(b_norm)] <= threshold


def token_set_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """Token-based matching — handles word reordering.
    
    "Republic of Korea" matches "Korea Republic" because
    they share the same content words.
    
    Args:
        a: first string
        b: second string
        threshold: minimum fraction of tokens that must match
        
    Returns:
        True if token overlap exceeds threshold
    """
    tokens_a = set(remove_stopwords(tokenize(a)))
    tokens_b = set(remove_stopwords(tokenize(b)))

    if not tokens_a or not tokens_b:
        return False

    # Jaccard similarity on token sets
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    if not union:
        return False

    jaccard = len(intersection) / len(union)
    return jaccard >= threshold


def abbreviation_match(a: str, b: str) -> bool:
    """Check if one string is an abbreviation of the other.
    
    "NYC" matches "New York City" (first letters of each word)
    "USA" matches "United States of America"
    
    Args:
        a: first string (could be abbreviation or full name)
        b: second string
        
    Returns:
        True if one is an abbreviation of the other
    """
    def get_initials(text: str) -> str:
        """Get first letter of each significant word."""
        tokens = remove_stopwords(tokenize(text))
        return "".join(t[0] for t in tokens if t)

    norm_a = normalize(a)
    norm_b = normalize(b)

    # Check if a is abbreviation of b
    initials_b = get_initials(norm_b)
    if norm_a == initials_b and len(norm_a) >= 2:
        return True

    # Check if b is abbreviation of a
    initials_a = get_initials(norm_a)
    if norm_b == initials_a and len(norm_b) >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Enhanced matcher combining all strategies
# ---------------------------------------------------------------------------

class EnhancedStringMatcher:
    """Combines multiple string matching strategies.
    
    Matching priority:
    1. Exact match (after normalization)
    2. Edit distance (paper's original approach)
    3. Token set matching (handles word reordering)
    4. Abbreviation detection (handles acronyms)
    """

    def __init__(
        self,
        use_edit_distance: bool = True,
        use_token_matching: bool = True,
        use_abbreviation: bool = True,
        edit_fed: float = 0.2,
        edit_ked: int = 10,
        token_threshold: float = 0.8,
    ):
        self.use_edit_distance = use_edit_distance
        self.use_token_matching = use_token_matching
        self.use_abbreviation = use_abbreviation
        self.edit_fed = edit_fed
        self.edit_ked = edit_ked
        self.token_threshold = token_threshold

    def are_match(self, a: str, b: str) -> Tuple[bool, str]:
        """Check if two strings match using any available strategy.
        
        Args:
            a: first string
            b: second string
            
        Returns:
            Tuple of (matched: bool, strategy: str)
        """
        if not a or not b:
            return False, "empty"

        # 1. Exact match after normalization
        if exact_match(a, b):
            return True, "exact"

        # 2. Edit distance (paper's original)
        if self.use_edit_distance:
            if edit_distance_match(a, b, self.edit_fed, self.edit_ked):
                return True, "edit_distance"

        # 3. Token set matching
        if self.use_token_matching:
            if token_set_match(a, b, self.token_threshold):
                return True, "token_set"

        # 4. Abbreviation detection
        if self.use_abbreviation:
            if abbreviation_match(a, b):
                return True, "abbreviation"

        return False, "no_match"

    def similarity_score(self, a: str, b: str) -> float:
        """Compute a similarity score between 0 and 1.
        
        Returns:
            1.0 for exact match
            0.8 for edit distance match
            0.7 for token set match
            0.6 for abbreviation match
            0.0 for no match
        """
        matched, strategy = self.are_match(a, b)
        if not matched:
            return 0.0
        scores = {
            "exact": 1.0,
            "edit_distance": 0.8,
            "token_set": 0.7,
            "abbreviation": 0.6,
        }
        return scores.get(strategy, 0.0)


# ---------------------------------------------------------------------------
# Comparison with original paper approach
# ---------------------------------------------------------------------------

def compare_with_original(test_pairs: List[Tuple[str, str]]) -> None:
    """Compare enhanced matcher with original edit distance approach."""
    matcher = EnhancedStringMatcher()

    print(f"\n  {'String A':<35} {'String B':<35} {'Original':>10} {'Enhanced':>10} {'Strategy'}")
    print(f"  {'-'*100}")

    for a, b in test_pairs:
        original = edit_distance_match(a, b)
        enhanced, strategy = matcher.are_match(a, b)
        orig_icon = "✓" if original else "✗"
        enh_icon = "✓" if enhanced else "✗"
        improvement = " ← NEW" if enhanced and not original else ""
        print(f"  {a:<35} {b:<35} {orig_icon:>10} {enh_icon:>10} {strategy}{improvement}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """Quick demo showing improvements over original edit distance."""
    print("Enhanced String Matcher — comparison with original approach\n")

    test_pairs = [
        # Edit distance handles these
        ("Korea, Republic of", "Korea Republic of"),
        ("American Samoa (US)", "American Samoa"),
        # Token matching handles these (word reorder)
        ("Republic of Korea", "Korea Republic"),
        ("United States Virgin Islands", "Virgin Islands United States"),
        # Abbreviation handles these
        ("NYC", "New York City"),
        ("USA", "United States of America"),
        ("UK", "United Kingdom"),
        # Punctuation normalization
        ("U.S.A.", "USA"),
        ("New York, NY", "New York NY"),
        # Should NOT match
        ("France", "Germany"),
        ("apple", "microsoft"),
    ]

    compare_with_original(test_pairs)


if __name__ == "__main__":
    demo()