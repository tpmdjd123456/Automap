"""Unit tests for WP3 (Table Synthesis — paper §4).

Covers approximate string matching, positive / negative compatibility
scores, the greedy partitioning algorithm, and output helpers.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

# Ensure repo root is on sys.path when running from tests/ sub-directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthesis import (
    approx_match,
    build_inverted_index,
    greedy_partition,
    load_candidates,
    negative_score,
    positive_score,
    save_synthesized_mappings,
    synthesize_mapping,
    synthesis_report,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_candidate(pairs):
    """Build a minimal candidate dict from a list of (l, r) tuples."""
    return {
        "pairs": [tuple(p) for p in pairs],
        "theta": 1.0,
        "row_count": len(pairs),
        "covered_rows": len(pairs),
        "source_table_index": 0,
        "left_column_index": 0,
        "right_column_index": 1,
        "source_metadata": {},
    }


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def paper_candidates():
    """Five candidates matching the paper's Figure 3 / Table 8 example.

    B1, B2 = IOC codes  (should merge — same system, different synonyms)
    B3, B4, B5 = ISO codes (should merge — same system)
    IOC and ISO must NOT merge (algeria: alg vs dza is a hard conflict).
    """
    return [
        # B1: IOC table 1
        {
            "pairs": [
                ("afghanistan", "afg"),
                ("albania", "alb"),
                ("algeria", "alg"),
                ("south korea", "kor"),
            ]
        },
        # B2: IOC table 2 (different synonym for south korea)
        {
            "pairs": [
                ("afghanistan", "afg"),
                ("albania", "alb"),
                ("algeria", "alg"),
                ("korea, republic of", "kor"),
            ]
        },
        # B3: ISO table 1
        {
            "pairs": [
                ("afghanistan", "afg"),
                ("albania", "alb"),
                ("algeria", "dza"),
                ("south korea", "kor"),
            ]
        },
        # B4: ISO table 2
        {
            "pairs": [
                ("germany", "deu"),
                ("france", "fra"),
                ("algeria", "dza"),
            ]
        },
        # B5: ISO table 3
        {
            "pairs": [
                ("japan", "jpn"),
                ("china", "chn"),
                ("algeria", "dza"),
            ]
        },
    ]


@pytest.fixture
def ioc_table():
    """Small IOC-style table for score tests."""
    return [
        ("afghanistan", "afg"),
        ("albania", "alb"),
        ("algeria", "alg"),
    ]


@pytest.fixture
def iso_table():
    """Small ISO-style table for score tests — conflicts on algeria."""
    return [
        ("afghanistan", "afg"),
        ("albania", "alb"),
        ("algeria", "dza"),
    ]


@pytest.fixture
def ioc_b1():
    """IOC table 1 — from paper Table 8a (richer, with more country entries)."""
    return make_candidate([
        ("afghanistan", "afg"), ("albania", "alb"),
        ("algeria", "alg"), ("south korea", "kor"),
        ("us virgin islands", "isv"),
    ])


@pytest.fixture
def ioc_b2():
    """IOC table 2 — from paper Table 8b (synonymous country names)."""
    return make_candidate([
        ("afghanistan", "afg"), ("albania", "alb"),
        ("algeria", "alg"), ("american samoa (us)", "asa"),
        ("korea, republic of (south)", "kor"),
        ("united states virgin islands", "isv"),
    ])


@pytest.fixture
def iso_b3():
    """ISO table — from paper Table 8c (different codes for some countries)."""
    return make_candidate([
        ("afghanistan", "afg"), ("albania", "alb"),
        ("algeria", "dza"),           # conflict: alg vs dza
        ("american samoa", "asm"),    # conflict: asa vs asm
        ("south korea", "kor"),
        ("us virgin islands", "vir"),  # conflict: isv vs vir
    ])


# ===========================================================================
# Approximate string matching
# ===========================================================================

def test_exact_match_scores_zero_distance():
    """Identical strings have edit distance 0 and always match."""
    assert approx_match("germany", "germany")
    assert approx_match("usa", "usa")
    assert approx_match("", "")


def test_short_strings_require_exact_match():
    """'usa' has len=3, threshold=floor(3*0.2)=0. 'usa' and 'rsa' must NOT match."""
    assert not approx_match("usa", "rsa")
    assert not approx_match("ab", "cd")


def test_approx_match_handles_minor_variation():
    """'american samoa' vs 'american samoa (us)' — edit distance <= threshold."""
    # len("american samoa") = 14, threshold = floor(14*0.2) = 2
    # "american samoa (us)" has 5 extra chars — too far; use a realistic pair
    # Try a pair that definitely fits within threshold
    # "organisation" vs "organization": edit distance 1
    # len=12, threshold=floor(12*0.2)=2 → should match
    assert approx_match("organisation", "organization")


def test_approx_match_long_strings_more_tolerance():
    """Longer strings allow proportionally larger edit distance."""
    # len("south korea") = 11, threshold = floor(11*0.2) = 2
    # "south korea" vs "south_korea": edit distance 1 (space vs underscore)
    assert approx_match("south korea", "south_korea")


def test_approx_match_symmetric():
    """approx_match(a, b) == approx_match(b, a) for various pairs."""
    pairs = [
        ("germany", "germany"),
        ("usa", "rsa"),
        ("organisation", "organization"),
        ("south korea", "south_korea"),
        ("", "hello"),
    ]
    for a, b in pairs:
        assert approx_match(a, b) == approx_match(b, a), (
            f"Symmetry failed for ({a!r}, {b!r})"
        )


def test_empty_string_does_not_match_nonempty():
    """'' vs 'germany': threshold=floor(0*0.2)=0, edit distance=7 → no match."""
    assert not approx_match("", "germany")
    assert not approx_match("germany", "")


def test_band_dp_matches_naive_edit_distance():
    """Band DP gives the same True/False result as a naive full-matrix check."""
    from synthesis import _naive_edit_distance

    test_pairs = [
        ("hello", "hello"),
        ("kitten", "sitting"),
        ("saturday", "sunday"),
        ("organisation", "organization"),
        ("usa", "rsa"),
        ("south korea", "south_korea"),
        ("germany", "germany"),
        ("", ""),
        ("", "x"),
        ("abc", "abcd"),
    ]
    for v1, v2 in test_pairs:
        threshold = min(
            math.floor(len(v1) * 0.2),
            math.floor(len(v2) * 0.2),
            10,
        )
        naive_dist = _naive_edit_distance(v1, v2)
        naive_result = naive_dist <= threshold
        band_result = approx_match(v1, v2)
        assert band_result == naive_result, (
            f"Mismatch for ({v1!r}, {v2!r}): "
            f"naive={naive_result} (dist={naive_dist}, thresh={threshold}), "
            f"band={band_result}"
        )


def test_different_long_strings_no_match():
    """Clearly different long strings must not match."""
    assert not approx_match("germany", "france")


def test_fed_parameter_respected():
    """With fed=0.0, threshold is always 0 — every pair requires exact match."""
    assert approx_match("germany", "germany", fed=0.0) is True
    assert approx_match("germany", "germny", fed=0.0) is False


def test_ked_cap_respected():
    """With ked=0, threshold is capped at 0 — every pair requires exact match."""
    assert approx_match("afghanistan", "afghnistan", ked=0) is False
    assert approx_match("afghanistan", "afghanistan", ked=0) is True


def test_empty_empty_approx_match_is_stable():
    """approx_match('', '') must not raise and must return a bool."""
    result = approx_match("", "")
    assert isinstance(result, bool)


# ===========================================================================
# Positive score
# ===========================================================================

def test_positive_score_identical_tables():
    """Two tables with the same pairs -> w+ = 1.0."""
    b = [("germany", "deu"), ("france", "fra"), ("japan", "jpn")]
    assert positive_score(b, b) == pytest.approx(1.0)


def test_positive_score_disjoint_tables():
    """Two tables with no shared pairs -> w+ = 0.0."""
    b1 = [("germany", "deu"), ("france", "fra")]
    b2 = [("msft", "microsoft"), ("aapl", "apple")]
    assert positive_score(b1, b2, use_approx=False) == pytest.approx(0.0)


def test_positive_score_partial_overlap():
    """B1 has 6 pairs, B2 has 6 pairs, 3 shared -> w+ = max(3/6, 3/6) = 0.5."""
    shared = [("a", "1"), ("b", "2"), ("c", "3")]
    only_b1 = [("d", "4"), ("e", "5"), ("f", "6")]
    only_b2 = [("g", "7"), ("h", "8"), ("i", "9")]
    b1 = shared + only_b1
    b2 = shared + only_b2
    assert positive_score(b1, b2, use_approx=False) == pytest.approx(0.5)


def test_positive_score_asymmetric_containment():
    """B1 is fully contained in B2: w+ = max(3/3, 3/10) = 1.0 (not 0.3)."""
    b1 = [("a", "1"), ("b", "2"), ("c", "3")]
    extra = [(str(i), str(i + 10)) for i in range(4, 11)]
    b2 = b1 + extra  # 10 pairs total
    result = positive_score(b1, b2, use_approx=False)
    assert result == pytest.approx(1.0)


def test_positive_score_with_approx_matching():
    """Approx matching: 'organisation' and 'organization' count as same left value."""
    b1 = [("organisation", "org")]
    b2 = [("organization", "org")]
    # With approx: should count as 1 shared pair
    assert positive_score(b1, b2, use_approx=True) == pytest.approx(1.0)
    # Without approx: different strings -> 0 shared
    assert positive_score(b1, b2, use_approx=False) == pytest.approx(0.0)


def test_positive_score_symmetric():
    """w+(B1, B2) == w+(B2, B1)."""
    b1 = [("germany", "deu"), ("france", "fra"), ("x", "y")]
    b2 = [("germany", "deu"), ("japan", "jpn")]
    assert positive_score(b1, b2) == pytest.approx(positive_score(b2, b1))


def test_positive_score_empty_inputs():
    """Empty tables yield 0.0 without raising."""
    b = [("germany", "deu")]
    assert positive_score([], b) == 0.0
    assert positive_score(b, []) == 0.0
    assert positive_score([], []) == 0.0


def test_approx_boosts_score(ioc_b1, ioc_b2):
    """IOC B1 and B2 share more matched pairs with approx than with exact."""
    b1 = ioc_b1["pairs"]
    b2 = ioc_b2["pairs"]
    exact_score = positive_score(b1, b2, use_approx=False)
    approx_s = positive_score(b1, b2, use_approx=True)
    assert approx_s >= exact_score


# ===========================================================================
# Negative score
# ===========================================================================

def test_negative_score_no_conflict(ioc_table):
    """Same IOC table vs itself: no conflict -> w- = 0.0."""
    assert negative_score(ioc_table, ioc_table) == pytest.approx(0.0)


def test_negative_score_full_conflict():
    """Every shared left value maps to a different right -> w- close to -1.0."""
    b1 = [("a", "x"), ("b", "y"), ("c", "z")]
    b2 = [("a", "p"), ("b", "q"), ("c", "r")]
    score = negative_score(b1, b2, use_approx=False)
    assert score == pytest.approx(-1.0)


def test_negative_score_partial_conflict():
    """B1: 3 pairs, B3: 3 pairs, 1 conflict (algeria). w- = -max(1/3, 1/3) = -0.333."""
    b1 = [("germany", "deu"), ("france", "fra"), ("algeria", "alg")]
    b3 = [("germany", "deu"), ("france", "fra"), ("algeria", "dza")]
    score = negative_score(b1, b3, use_approx=False)
    assert score == pytest.approx(-1 / 3, abs=1e-9)


def test_ioc_vs_iso_correctly_negative(ioc_table, iso_table):
    """IOC and ISO tables conflict on 'algeria': w- = -max(1/3, 1/3) = -0.333."""
    score = negative_score(ioc_table, iso_table, use_approx=False)
    assert score == pytest.approx(-1 / 3, abs=1e-9)
    assert score < 0


def test_negative_score_symmetric():
    """w-(B1, B2) == w-(B2, B1)."""
    b1 = [("germany", "deu"), ("france", "fra"), ("algeria", "alg")]
    b2 = [("germany", "deu"), ("france", "fra"), ("algeria", "dza")]
    assert negative_score(b1, b2) == pytest.approx(negative_score(b2, b1))


def test_negative_score_empty_inputs():
    """Empty tables yield 0.0 without raising."""
    b = [("germany", "deu")]
    assert negative_score([], b) == 0.0
    assert negative_score(b, []) == 0.0


def test_disjoint_left_values_no_conflict():
    """Tables with completely different left values produce no conflicts."""
    b1 = [("germany", "deu"), ("france", "fra")]
    b2 = [("japan", "jpn"), ("china", "chn")]
    assert negative_score(b1, b2, use_approx=False) == pytest.approx(0.0)


# ===========================================================================
# Greedy partitioning
# ===========================================================================

def _partition_map(partitions):
    """Return dict mapping candidate_index -> partition_index."""
    return {idx: i for i, p in enumerate(partitions) for idx in p}


def test_ioc_tables_merged(paper_candidates):
    """B1 (idx 0) and B2 (idx 1) are both IOC — must end up in the same partition."""
    partitions = greedy_partition(paper_candidates)
    flat = _partition_map(partitions)
    assert flat[0] == flat[1], "IOC tables B1 and B2 must merge"


def test_iso_tables_merged(paper_candidates):
    """B3, B4, B5 (idx 2,3,4) are all ISO — must be in the same partition."""
    partitions = greedy_partition(paper_candidates)
    flat = _partition_map(partitions)
    assert flat[2] == flat[3] == flat[4], "ISO tables B3, B4, B5 must merge"


def test_ioc_and_iso_separated(paper_candidates):
    """IOC and ISO partitions must be different (algeria conflict blocks merge)."""
    partitions = greedy_partition(paper_candidates)
    flat = _partition_map(partitions)
    assert flat[0] != flat[2], "IOC and ISO must NOT merge"


def test_all_candidates_assigned(paper_candidates):
    """Every candidate index appears in exactly one partition."""
    partitions = greedy_partition(paper_candidates)
    all_indices = sorted(idx for p in partitions for idx in p)
    assert all_indices == list(range(len(paper_candidates)))


def test_greedy_produces_disjoint_partitions(paper_candidates):
    """No candidate index appears in more than one partition."""
    partitions = greedy_partition(paper_candidates)
    all_indices = [idx for p in partitions for idx in p]
    assert len(all_indices) == len(set(all_indices))


def test_greedy_covers_all_candidates(paper_candidates):
    """Union of all partitions equals the set of all candidate indices."""
    partitions = greedy_partition(paper_candidates)
    all_indices = {idx for p in partitions for idx in p}
    assert all_indices == set(range(len(paper_candidates)))


def test_singleton_when_no_overlap():
    """A candidate that shares nothing with others stays as its own partition."""
    candidates = [
        {"pairs": [("germany", "deu"), ("france", "fra")]},
        {"pairs": [("msft", "microsoft"), ("aapl", "apple")]},
    ]
    partitions = greedy_partition(candidates, use_approx=False)
    assert len(partitions) == 2


def test_tau_blocks_conflicting_merge():
    """With tau=-0.2, a pair with w-=-0.5 must NOT be merged."""
    # b1 and b2 share 1 pair out of 2 but conflict on "a"
    # F = {"a"} → w- = -max(1/2, 1/2) = -0.5 < tau=-0.2 → blocked
    candidates = [
        {"pairs": [("a", "x"), ("b", "y")]},
        {"pairs": [("a", "z"), ("b", "y")]},
    ]
    partitions = greedy_partition(candidates, tau=-0.2, use_approx=False)
    assert len(partitions) == 2, "Merge must be blocked by w- < tau"


def test_tau_zero_is_stricter():
    """With tau=0.0, any negative score (any conflict) blocks the merge."""
    candidates = [
        {"pairs": [("a", "x"), ("b", "y"), ("c", "z")]},
        {"pairs": [("a", "p"), ("b", "y"), ("c", "z")]},
    ]
    # F = {"a"}: w- = -1/3 ≈ -0.333 < 0.0 → blocked
    partitions = greedy_partition(candidates, tau=0.0, use_approx=False)
    assert len(partitions) == 2, "Merge must be blocked with tau=0.0"


def test_tau_allows_minor_conflict():
    """With tau=-0.5, a pair with w-=-1/3 should still be allowed to merge."""
    candidates = [
        {"pairs": [("a", "x"), ("b", "y"), ("c", "z")]},
        {"pairs": [("a", "p"), ("b", "y"), ("c", "z")]},
    ]
    # w- = -1/3 ≈ -0.333 >= -0.5 → merge is allowed
    partitions = greedy_partition(candidates, tau=-0.5, use_approx=False)
    assert len(partitions) == 1


def test_tau_default_allows_merge_when_no_conflict():
    """Two compatible candidates with no conflict must merge at tau=-0.2."""
    candidates = [
        make_candidate([("germany", "deu"), ("france", "fra")]),
        make_candidate([("germany", "deu"), ("japan", "jpn")]),
    ]
    partitions = greedy_partition(candidates, tau=-0.2, use_approx=False)
    assert len(partitions) == 1


def test_single_candidate_returns_one_partition():
    """A single candidate produces exactly one partition containing index 0."""
    candidates = [make_candidate([("germany", "deu")])]
    partitions = greedy_partition(candidates)
    assert len(partitions) == 1
    assert partitions[0] == [0]


def test_empty_candidates_returns_empty():
    """greedy_partition on an empty list returns an empty list."""
    assert greedy_partition([]) == []


def test_three_way_merge():
    """Three tables that all agree on values should all end up in one partition."""
    candidates = [
        make_candidate([("a", "1"), ("b", "2")]),
        make_candidate([("a", "1"), ("c", "3")]),
        make_candidate([("b", "2"), ("c", "3")]),
    ]
    partitions = greedy_partition(candidates, tau=-0.2, use_approx=False)
    flat = _partition_map(partitions)
    assert flat[0] == flat[1] == flat[2]


def test_two_independent_groups():
    """Two distinct groups merge within themselves but not between each other."""
    candidates = [
        make_candidate([("germany", "deu"), ("france", "fra")]),
        make_candidate([("germany", "deu"), ("italy", "ita")]),
        make_candidate([("msft", "microsoft"), ("aapl", "apple")]),
        make_candidate([("msft", "microsoft"), ("googl", "alphabet")]),
    ]
    partitions = greedy_partition(candidates, tau=-0.2, use_approx=False)
    flat = _partition_map(partitions)
    assert flat[0] == flat[1]
    assert flat[2] == flat[3]
    assert flat[0] != flat[2]


# ===========================================================================
# Inverted index
# ===========================================================================

def test_pair_index_maps_pair_to_candidates():
    """pair_index[('germany', 'deu')] lists all candidates containing that pair."""
    candidates = [
        {"pairs": [("germany", "deu"), ("france", "fra")]},
        {"pairs": [("germany", "deu"), ("japan", "jpn")]},
        {"pairs": [("usa", "united states")]},
    ]
    pair_index, _ = build_inverted_index(candidates)
    assert set(pair_index[("germany", "deu")]) == {0, 1}
    assert ("usa", "united states") in pair_index


def test_left_index_maps_left_to_candidates():
    """left_index['algeria'] lists all candidates that have 'algeria' on the left."""
    candidates = [
        {"pairs": [("algeria", "alg"), ("france", "fra")]},
        {"pairs": [("algeria", "dza"), ("germany", "deu")]},
        {"pairs": [("japan", "jpn")]},
    ]
    _, left_index = build_inverted_index(candidates)
    assert set(left_index["algeria"]) == {0, 1}
    assert "japan" in left_index
    assert 2 in left_index["japan"]


def test_inverted_index_reduces_comparisons():
    """Candidates with completely different domains share no key in either index."""
    candidates = [
        {"pairs": [("germany", "deu"), ("france", "fra")]},
        {"pairs": [("msft", "microsoft"), ("aapl", "apple")]},
    ]
    pair_index, left_index = build_inverted_index(candidates)
    # No key should list both index 0 and index 1
    for indices in pair_index.values():
        assert not ({0, 1} <= set(indices))
    for indices in left_index.values():
        assert not ({0, 1} <= set(indices))


# ===========================================================================
# Synthesis and output helpers
# ===========================================================================

def test_synthesize_mapping_unions_pairs():
    """Union of two candidates with partial overlap contains all pairs."""
    candidates = [
        {"pairs": [("germany", "deu"), ("france", "fra")]},
        {"pairs": [("france", "fra"), ("japan", "jpn")]},
    ]
    result = synthesize_mapping([0, 1], candidates)
    assert ("germany", "deu") in result
    assert ("france", "fra") in result
    assert ("japan", "jpn") in result


def test_synthesize_mapping_deduplicates():
    """Pairs appearing in multiple candidates appear only once in the output."""
    candidates = [
        {"pairs": [("a", "1"), ("b", "2")]},
        {"pairs": [("a", "1"), ("c", "3")]},
    ]
    result = synthesize_mapping([0, 1], candidates)
    assert result.count(("a", "1")) == 1


def test_synthesize_mapping_sorted():
    """Output is sorted lexicographically."""
    candidates = [
        {"pairs": [("z", "9"), ("a", "1"), ("m", "5")]},
    ]
    result = synthesize_mapping([0], candidates)
    assert result == sorted(result)


def test_save_synthesized_mappings_jsonl_schema(tmp_path, paper_candidates):
    """Each line of synthesized_mappings.jsonl has the 5 required fields."""
    partitions = greedy_partition(paper_candidates)
    output = str(tmp_path / "out.jsonl")
    save_synthesized_mappings(partitions, paper_candidates, output)
    with open(output, encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == len(partitions)
    required = {"partition_id", "candidate_indices", "pairs", "size", "num_source_tables"}
    for line in lines:
        rec = json.loads(line)
        assert required <= rec.keys(), f"Missing fields in: {rec}"
        assert isinstance(rec["partition_id"], int)
        assert isinstance(rec["candidate_indices"], list)
        assert isinstance(rec["pairs"], list)
        assert rec["size"] == len(rec["pairs"])
        assert rec["num_source_tables"] == len(rec["candidate_indices"])


def test_save_load_roundtrip(tmp_path, paper_candidates):
    """save_synthesized_mappings then reading back produces consistent data."""
    partitions = greedy_partition(paper_candidates)
    output = str(tmp_path / "mappings.jsonl")
    save_synthesized_mappings(partitions, paper_candidates, output)
    lines = open(output, encoding="utf-8").readlines()
    assert len(lines) == len(partitions)
    # Verify partition_id is sequential
    for expected_id, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["partition_id"] == expected_id


def test_synthesis_report_runs_without_error(capsys, paper_candidates):
    """synthesis_report() prints without raising exceptions."""
    partitions = greedy_partition(paper_candidates)
    synthesis_report(partitions, paper_candidates)
    captured = capsys.readouterr()
    assert "Total synthesized mappings" in captured.out
    assert "Singleton" in captured.out


def test_synthesis_report_handles_empty(capsys):
    """synthesis_report() on empty input prints a graceful message."""
    synthesis_report([], [])
    out = capsys.readouterr().out
    assert "No partitions" in out


def test_save_creates_parent_directories(tmp_path):
    """save_synthesized_mappings creates any missing parent directories."""
    candidates = [make_candidate([("a", "1")])]
    partitions = [[0]]
    nested = str(tmp_path / "nested" / "sub" / "out.jsonl")
    save_synthesized_mappings(partitions, candidates, nested)
    assert os.path.exists(nested)


def test_load_candidates_roundtrip(tmp_path):
    """load_candidates converts pairs to list-of-tuples."""
    import json as _json

    records = [
        {
            "pairs": [["germany", "deu"], ["france", "fra"]],
            "theta": 1.0,
            "row_count": 2,
            "covered_rows": 2,
            "source_table_index": 0,
            "left_column_index": 0,
            "right_column_index": 1,
            "source_metadata": {},
        }
    ]
    path = str(tmp_path / "candidates.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(_json.dumps(r) + "\n")

    loaded = load_candidates(path)
    assert len(loaded) == 1
    pairs = loaded[0]["pairs"]
    assert all(isinstance(p, tuple) for p in pairs)
    assert pairs[0] == ("germany", "deu")
