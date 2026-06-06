"""Jaccard Similarity String Matching — alternative to edit distance.

Jaccard similarity measures the overlap between two sets of n-grams
(character chunks). Unlike edit distance which counts character changes,
Jaccard measures how much two strings share in common.

Particularly effective for:
- Strings with same words in different order
- Strings with shared substrings
- Longer entity names

Comparison with other approaches:
- Edit Distance: counts character changes (position matters)
- Jaro-Winkler: measures character overlap with prefix bonus
- Jaccard: measures n-gram set overlap (position doesn't matter)

Paper reference: This is an original contribution extending §4.1
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import FrozenSet, List, Set, Tuple


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=500_000)
def normalize(text: str) -> str:
    """Normalize text before comparison.

    Cached: in WP3 scoring the same string value is normalized millions of
    times across candidate pair comparisons. lru_cache keys on the input
    string and returns the cached normalized form.
    """
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# N-gram generation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=500_000)
def char_ngrams(text: str, n: int = 2) -> FrozenSet[str]:
    """Generate character n-grams from text.

    Returns a frozenset (immutable) so the cached value is safe to share
    across many calls. WP3 scoring calls this with the same strings
    millions of times; caching is the single biggest speedup.
    """
    text = normalize(text)
    if len(text) < n:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i:i + n] for i in range(len(text) - n + 1))


@lru_cache(maxsize=500_000)
def word_ngrams(text: str, n: int = 1) -> FrozenSet[str]:
    """Generate word n-grams from text.

    Cached frozenset, same rationale as ``char_ngrams``.
    """
    words = normalize(text).split()
    if n == 1:
        return frozenset(words)
    if len(words) < n:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets.
    
    Jaccard = |A ∩ B| / |A ∪ B|
    
    Args:
        set_a: first set
        set_b: second set
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def char_jaccard(a: str, b: str, n: int = 2) -> float:
    """Compute Jaccard similarity using character n-grams.
    
    Args:
        a: first string
        b: second string
        n: n-gram size
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    return jaccard(char_ngrams(a, n), char_ngrams(b, n))


def word_jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity using words.
    
    Args:
        a: first string
        b: second string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    return jaccard(word_ngrams(a), word_ngrams(b))


# ---------------------------------------------------------------------------
# JaccardMatcher class
# ---------------------------------------------------------------------------

class JaccardMatcher:
    """String matcher using Jaccard similarity.
    
    Combines character n-gram Jaccard and word Jaccard
    for robust string matching.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ngram_size: int = 2,
        use_word_jaccard: bool = True,
        use_char_jaccard: bool = True,
    ):
        """Initialize matcher.
        
        Args:
            threshold: minimum Jaccard score to consider a match
            ngram_size: size of character n-grams
            use_word_jaccard: use word-level Jaccard
            use_char_jaccard: use character n-gram Jaccard
        """
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.use_word_jaccard = use_word_jaccard
        self.use_char_jaccard = use_char_jaccard

    def similarity(self, a: str, b: str) -> float:
        """Compute combined Jaccard similarity.
        
        Takes the maximum of word and character Jaccard scores.
        
        Args:
            a: first string
            b: second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        scores = []
        if self.use_char_jaccard:
            scores.append(char_jaccard(a, b, self.ngram_size))
        if self.use_word_jaccard:
            scores.append(word_jaccard(a, b))
        return max(scores) if scores else 0.0

    def are_match(self, a: str, b: str) -> Tuple[bool, float]:
        """Check if two strings match using Jaccard similarity.
        
        Args:
            a: first string
            b: second string
            
        Returns:
            Tuple of (matched: bool, score: float)
        """
        if not a or not b:
            return False, 0.0
        score = self.similarity(a, b)
        return score >= self.threshold, round(score, 4)


# ---------------------------------------------------------------------------
# Comparison with edit distance and Jaro-Winkler
# ---------------------------------------------------------------------------

def compare_all(test_pairs: List[Tuple[str, str]]) -> None:
    """Compare Jaccard with edit distance and Jaro-Winkler."""
    from string_matcher import edit_distance_match
    from jaro_winkler import JaroWinklerMatcher

    jw_matcher = JaroWinklerMatcher(threshold=0.85)
    jac_matcher = JaccardMatcher(threshold=0.5)

    print(f"\n  {'String A':<32} {'String B':<32} {'ED':>5} {'JW':>5} {'JAC':>5} {'JAC Score':>10}")
    print(f"  {'-'*95}")

    ed_total = jw_total = jac_total = 0

    for a, b in test_pairs:
        ed = edit_distance_match(a, b)
        jw, jw_score = jw_matcher.are_match(a, b)
        jac, jac_score = jac_matcher.are_match(a, b)

        ed_icon  = "✓" if ed  else "✗"
        jw_icon  = "✓" if jw  else "✗"
        jac_icon = "✓" if jac else "✗"

        if ed:  ed_total  += 1
        if jw:  jw_total  += 1
        if jac: jac_total += 1

        print(f"  {a:<32} {b:<32} {ed_icon:>5} {jw_icon:>5} {jac_icon:>5} {jac_score:>10}")

    print(f"\n  Summary ({len(test_pairs)} test pairs):")
    print(f"  Edit Distance : {ed_total}/{len(test_pairs)}")
    print(f"  Jaro-Winkler  : {jw_total}/{len(test_pairs)}")
    print(f"  Jaccard       : {jac_total}/{len(test_pairs)}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """Compare all three matchers."""
    print("Jaccard vs Jaro-Winkler vs Edit Distance\n")

    test_pairs = [
        # Edit distance handles these
        ("Korea, Republic of",           "Korea Republic of"),
        ("American Samoa",               "American Samoa (US)"),
        # Jaro-Winkler handles these
        ("Microsoft Corp",               "Microsoft Corporation"),
        ("colour",                       "color"),
        # Jaccard handles these (word reordering)
        ("Republic of Korea",            "Korea Republic"),
        ("City of New York",             "New York City"),
        ("United States of America",     "America United States"),
        ("North Atlantic Treaty Org",    "Treaty Org North Atlantic"),
        # All should handle
        ("Smith",                        "Smyth"),
        ("France",                       "France"),
        # Should NOT match
        ("France",                       "Germany"),
        ("apple",                        "microsoft"),
        ("New York",                     "Los Angeles"),
    ]

    compare_all(test_pairs)


if __name__ == "__main__":
    demo()