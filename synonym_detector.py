"""Synonym Detection — improvement to WP3 Table Synthesis.

The paper (§4.1) mentions that synonym feeds can be used to boost
positive compatibility between tables. For example, knowing that
"US Virgin Islands" and "United States Virgin Islands" are synonyms
allows tables containing either form to be correctly merged.

This module provides:
1. A built-in set of common synonyms as a starting point
2. Ability to load custom synonyms from a CSV file
3. A simple API for checking if two values are synonyms
4. Integration with the synthesis compatibility scoring

Usage:
    from synonym_detector import SynonymDetector
    sd = SynonymDetector()
    sd.load_from_csv("synonyms.csv")
    sd.are_synonyms("USA", "United States")  # True
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Built-in synonym pairs (generic, domain-independent)
# ---------------------------------------------------------------------------

BUILTIN_SYNONYMS: List[Tuple[str, str]] = [
    # Country name variations
    ("usa", "united states"),
    ("usa", "united states of america"),
    ("united states", "united states of america"),
    ("uk", "united kingdom"),
    ("uk", "great britain"),
    ("united kingdom", "great britain"),
    ("uae", "united arab emirates"),
    ("south korea", "korea republic"),
    ("south korea", "republic of korea"),
    ("north korea", "democratic peoples republic of korea"),
    ("russia", "russian federation"),
    ("iran", "islamic republic of iran"),
    ("syria", "syrian arab republic"),
    ("tanzania", "united republic of tanzania"),
    ("moldova", "republic of moldova"),
    ("bolivia", "plurinational state of bolivia"),
    ("venezuela", "bolivarian republic of venezuela"),
    ("congo", "democratic republic of the congo"),
    ("congo", "republic of the congo"),

    # Common abbreviations
    ("dr", "doctor"),
    ("mr", "mister"),
    ("mrs", "missus"),
    ("prof", "professor"),
    ("st", "saint"),
    ("mt", "mount"),
    ("ft", "fort"),

    # US state abbreviations
    ("al", "alabama"), ("ak", "alaska"), ("az", "arizona"),
    ("ar", "arkansas"), ("ca", "california"), ("co", "colorado"),
    ("ct", "connecticut"), ("de", "delaware"), ("fl", "florida"),
    ("ga", "georgia"), ("hi", "hawaii"), ("id", "idaho"),
    ("il", "illinois"), ("in", "indiana"), ("ia", "iowa"),
    ("ks", "kansas"), ("ky", "kentucky"), ("la", "louisiana"),
    ("me", "maine"), ("md", "maryland"), ("ma", "massachusetts"),
    ("mi", "michigan"), ("mn", "minnesota"), ("ms", "mississippi"),
    ("mo", "missouri"), ("mt", "montana"), ("ne", "nebraska"),
    ("nv", "nevada"), ("nh", "new hampshire"), ("nj", "new jersey"),
    ("nm", "new mexico"), ("ny", "new york"), ("nc", "north carolina"),
    ("nd", "north dakota"), ("oh", "ohio"), ("ok", "oklahoma"),
    ("or", "oregon"), ("pa", "pennsylvania"), ("ri", "rhode island"),
    ("sc", "south carolina"), ("sd", "south dakota"), ("tn", "tennessee"),
    ("tx", "texas"), ("ut", "utah"), ("vt", "vermont"),
    ("va", "virginia"), ("wa", "washington"), ("wv", "west virginia"),
    ("wi", "wisconsin"), ("wy", "wyoming"),

    # Music genre variations
    ("hip hop", "hip-hop"),
    ("r&b", "rhythm and blues"),
    ("rnb", "rhythm and blues"),
    ("rock and roll", "rock & roll"),
    ("indie", "independent"),

    # General variations
    ("and", "&"),
    ("intl", "international"),
    ("natl", "national"),
    ("corp", "corporation"),
    ("inc", "incorporated"),
    ("ltd", "limited"),
    ("dept", "department"),
    ("univ", "university"),
    ("assoc", "association"),
]


# ---------------------------------------------------------------------------
# SynonymDetector class
# ---------------------------------------------------------------------------

class SynonymDetector:
    """Detects synonyms between values using a lookup table.
    
    Synonyms are stored as equivalence classes — if A=B and B=C
    then A=C automatically (transitive closure).
    """

    def __init__(self, use_builtins: bool = True):
        """Initialize with optional built-in synonyms.
        
        Args:
            use_builtins: if True, load the built-in synonym pairs
        """
        # Maps each value to its canonical (representative) form
        self._canonical: Dict[str, str] = {}
        # Maps each canonical form to all its synonyms
        self._groups: Dict[str, Set[str]] = defaultdict(set)
        self._num_pairs = 0

        if use_builtins:
            for a, b in BUILTIN_SYNONYMS:
                self.add_synonym(a.lower(), b.lower())

    def add_synonym(self, a: str, b: str) -> None:
        """Add a synonym pair (a, b).
        
        Both values are normalized to lowercase.
        Uses union-find style merging for transitive closure.
        """
        a = a.lower().strip()
        b = b.lower().strip()
        if a == b:
            return

        canon_a = self._canonical.get(a, a)
        canon_b = self._canonical.get(b, b)

        if canon_a == canon_b:
            return  # Already in same group

        # Merge smaller group into larger group
        group_a = self._groups[canon_a] | {a, canon_a}
        group_b = self._groups[canon_b] | {b, canon_b}

        if len(group_a) >= len(group_b):
            canonical = canon_a
            merged = group_a | group_b
        else:
            canonical = canon_b
            merged = group_a | group_b

        # Update canonical mapping for all members
        for member in merged:
            self._canonical[member] = canonical
        self._groups[canonical] = merged
        self._num_pairs += 1

    def are_synonyms(self, a: str, b: str) -> bool:
        """Check if two values are synonyms.
        
        Args:
            a: first value
            b: second value
            
        Returns:
            True if a and b are known synonyms
        """
        a = a.lower().strip()
        b = b.lower().strip()
        if a == b:
            return True
        canon_a = self._canonical.get(a, a)
        canon_b = self._canonical.get(b, b)
        return canon_a == canon_b

    def get_synonyms(self, value: str) -> Set[str]:
        """Get all known synonyms for a value.
        
        Args:
            value: the value to look up
            
        Returns:
            Set of all synonyms including the value itself
        """
        value = value.lower().strip()
        canon = self._canonical.get(value, value)
        group = self._groups.get(canon, set())
        return group | {value}

    def load_from_csv(self, path: str) -> int:
        """Load synonym pairs from a CSV file.
        
        CSV format: two columns, no header required.
        Each row is one synonym pair: value_a, value_b
        
        Args:
            path: path to CSV file
            
        Returns:
            Number of synonym pairs loaded
        """
        if not os.path.exists(path):
            print(f"  Warning: synonym file not found: {path}")
            return 0

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    a, b = row[0].strip(), row[1].strip()
                    if a and b:
                        self.add_synonym(a, b)
                        count += 1
        print(f"  Loaded {count} synonym pairs from {path}")
        return count

    def save_to_csv(self, path: str) -> None:
        """Save all synonym pairs to a CSV file.
        
        Args:
            path: path to save CSV file
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        written = set()
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for canon, group in self._groups.items():
                members = sorted(group)
                for i, a in enumerate(members):
                    for b in members[i+1:]:
                        key = (min(a,b), max(a,b))
                        if key not in written:
                            writer.writerow([a, b])
                            written.add(key)
        print(f"  Saved synonyms to {path}")

    def summary(self) -> None:
        """Print a summary of loaded synonyms."""
        total_values = len(self._canonical)
        total_groups = len(self._groups)
        print(f"  Synonym detector summary:")
        print(f"  Total values with synonyms : {total_values}")
        print(f"  Total synonym groups       : {total_groups}")
        print(f"  Built-in pairs loaded      : {len(BUILTIN_SYNONYMS)}")
        if total_groups > 0:
            sizes = [len(g) for g in self._groups.values()]
            print(f"  Avg group size             : {sum(sizes)/len(sizes):.1f}")
            print(f"  Largest group              : {max(sizes)} synonyms")


# ---------------------------------------------------------------------------
# Integration with synthesis compatibility scoring
# ---------------------------------------------------------------------------

def boost_compatibility_with_synonyms(
    pairs_a: List[Tuple[str, str]],
    pairs_b: List[Tuple[str, str]],
    detector: SynonymDetector,
) -> int:
    """Count additional matches between two tables using synonym detection.
    
    When two tables don't share exact value pairs but share synonym pairs,
    this function counts those as additional matches to boost compatibility.
    
    Args:
        pairs_a: value pairs from first table
        pairs_b: value pairs from second table
        detector: SynonymDetector instance
        
    Returns:
        Number of additional synonym-based matches found
    """
    # Get exact matches first (already counted in synthesis)
    exact_a = set(pairs_a)
    exact_b = set(pairs_b)
    exact_matches = exact_a & exact_b

    # Find synonym matches not already in exact matches
    synonym_matches = 0
    for la, ra in pairs_a:
        for lb, rb in pairs_b:
            if (la, ra) in exact_matches:
                continue
            if (lb, rb) in exact_matches:
                continue
            # Check if left values are synonyms AND right values are synonyms
            if detector.are_synonyms(la, lb) and detector.are_synonyms(ra, rb):
                synonym_matches += 1

    return synonym_matches


# ---------------------------------------------------------------------------
# Demo / quick test
# ---------------------------------------------------------------------------

def demo() -> None:
    """Quick demo of synonym detection."""
    sd = SynonymDetector()
    sd.summary()

    print("\n  Example synonym checks:")
    tests = [
        ("USA", "United States"),
        ("USA", "United States of America"),
        ("UK", "United Kingdom"),
        ("South Korea", "Republic of Korea"),
        ("CA", "California"),
        ("hip hop", "hip-hop"),
        ("r&b", "rhythm and blues"),
        ("France", "Germany"),  # Not synonyms
        ("Inc", "Incorporated"),
    ]
    for a, b in tests:
        result = sd.are_synonyms(a, b)
        icon = "✓" if result else "✗"
        print(f"    {icon} '{a}' ↔ '{b}': {result}")


if __name__ == "__main__":
    demo()