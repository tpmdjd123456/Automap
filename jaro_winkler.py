"""Jaro-Winkler String Matching — alternative to edit distance.

Jaro-Winkler is specifically designed for short string matching
like entity names. It gives extra weight to strings that share
a common prefix, making it particularly effective for:
- Person names: "Smith" vs "Smyth"
- Place names: "Korea Republic" vs "Korea, Republic"
- Organization names: "Microsoft Corp" vs "Microsoft Corporation"

Comparison with edit distance:
- Edit distance: counts character insertions/deletions/substitutions
- Jaro-Winkler: measures character overlap and transpositions

Paper reference: This is an original contribution extending §4.1
"""

from __future__ import annotations

import re
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Text normalization (same as string_matcher.py)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text before comparison."""
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Jaro similarity
# ---------------------------------------------------------------------------

def jaro(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings.
    
    Jaro similarity is based on:
    - Number of matching characters
    - Number of transpositions
    
    Args:
        s1: first string
        s2: second string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if s1 == s2:
        return 1.0

    len_s1 = len(s1)
    len_s2 = len(s2)

    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    # Maximum distance for matching characters
    match_distance = max(len_s1, len_s2) // 2 - 1
    match_distance = max(0, match_distance)

    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    matches = 0
    transpositions = 0

    # Find matching characters
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len_s1 +
        matches / len_s2 +
        (matches - transpositions / 2) / matches
    ) / 3


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity
# ---------------------------------------------------------------------------

def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity between two strings.
    
    Jaro-Winkler extends Jaro by giving extra weight to strings
    that share a common prefix. This is particularly useful for
    entity names where the beginning of the name is most distinctive.
    
    Args:
        s1: first string
        s2: second string
        p: prefix weight (default 0.1, max 0.25)
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    jaro_score = jaro(s1, s2)

    # Find common prefix length (max 4 characters)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro_score + prefix * p * (1 - jaro_score)


# ---------------------------------------------------------------------------
# JaroWinklerMatcher class
# ---------------------------------------------------------------------------

class JaroWinklerMatcher:
    """String matcher using Jaro-Winkler similarity.
    
    Alternative to edit distance for entity name matching.
    Particularly effective for:
    - Names with typos: "Smyth" vs "Smith"
    - Names with punctuation: "Korea, Republic" vs "Korea Republic"
    - Names with minor variations: "Microsoft Corp" vs "Microsoft Corporation"
    """

    def __init__(
        self,
        threshold: float = 0.85,
        prefix_weight: float = 0.1,
    ):
        """Initialize matcher.
        
        Args:
            threshold: minimum Jaro-Winkler score to consider a match
            prefix_weight: weight given to common prefix (default 0.1)
        """
        self.threshold = threshold
        self.prefix_weight = prefix_weight

    def similarity(self, a: str, b: str) -> float:
        """Compute Jaro-Winkler similarity after normalization.
        
        Args:
            a: first string
            b: second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        a_norm = normalize(a)
        b_norm = normalize(b)
        return jaro_winkler(a_norm, b_norm, self.prefix_weight)

    def are_match(self, a: str, b: str) -> Tuple[bool, float]:
        """Check if two strings match using Jaro-Winkler.
        
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
# Comparison with edit distance
# ---------------------------------------------------------------------------

def compare_with_edit_distance(
    test_pairs: List[Tuple[str, str]],
    jw_threshold: float = 0.85,
) -> None:
    """Compare Jaro-Winkler with edit distance on test pairs."""
    from string_matcher import edit_distance_match

    matcher = JaroWinklerMatcher(threshold=jw_threshold)

    print(f"\n  {'String A':<35} {'String B':<30} {'Edit Dist':>10} {'Jaro-W':>10} {'Score':>8}")
    print(f"  {'-'*95}")

    ed_matches = 0
    jw_matches = 0
    jw_only = 0
    ed_only = 0

    for a, b in test_pairs:
        ed = edit_distance_match(a, b)
        jw, score = matcher.are_match(a, b)

        ed_icon = "✓" if ed else "✗"
        jw_icon = "✓" if jw else "✗"

        label = ""
        if jw and not ed:
            label = " ← JW only"
            jw_only += 1
        elif ed and not jw:
            label = " ← ED only"
            ed_only += 1

        if ed: ed_matches += 1
        if jw: jw_matches += 1

        print(f"  {a:<35} {b:<30} {ed_icon:>10} {jw_icon:>10} {score:>8}{label}")

    print(f"\n  Summary:")
    print(f"  Edit Distance matches : {ed_matches}/{len(test_pairs)}")
    print(f"  Jaro-Winkler matches  : {jw_matches}/{len(test_pairs)}")
    print(f"  JW finds but ED misses: {jw_only}")
    print(f"  ED finds but JW misses: {ed_only}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """Compare Jaro-Winkler with edit distance."""
    print("Jaro-Winkler vs Edit Distance Comparison\n")

    test_pairs = [
        # Edit distance handles these well
        ("Korea, Republic of",    "Korea Republic of"),
        ("American Samoa",        "American Samoa (US)"),
        # Jaro-Winkler should handle these better
        ("Microsoft Corp",        "Microsoft Corporation"),
        ("Korea Republic",        "Korea, Republic"),
        ("New York",              "New York City"),
        ("United States",         "United States of America"),
        ("Smith",                 "Smyth"),
        ("Johnathan",             "Jonathan"),
        ("colour",                "color"),
        ("centre",                "center"),
        # Should NOT match
        ("France",                "Germany"),
        ("apple",                 "microsoft"),
        ("New York",              "Los Angeles"),
    ]

    compare_with_edit_distance(test_pairs)


if __name__ == "__main__":
    demo()