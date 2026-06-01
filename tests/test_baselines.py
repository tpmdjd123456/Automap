"""Tests for baselines.py"""

import pytest
from baselines import (
    get_column_signature,
    get_domain,
    merge_tables,
    union_domain,
    union_web,
    schema_cc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_corpus():
    """Small corpus for testing baselines."""
    return [
        # Page 1, 2 columns
        {"pgId": 1, "relation": [["france", "germany"], ["fra", "deu"]]},
        {"pgId": 1, "relation": [["france", "japan"], ["fra", "jpn"]]},
        # Page 2, 2 columns
        {"pgId": 2, "relation": [["france", "italy"], ["fra", "ita"]]},
        # Page 1, 3 columns — different signature
        {"pgId": 1, "relation": [["france"], ["fra"], ["europe"]]},
        # Empty table
        {"pgId": 1, "relation": []},
    ]


# ---------------------------------------------------------------------------
# Tests for helpers
# ---------------------------------------------------------------------------

def test_get_column_signature():
    table = {"relation": [["a", "b"], ["c", "d"]]}
    assert get_column_signature(table) == "ncols_2"


def test_get_column_signature_three():
    table = {"relation": [["a"], ["b"], ["c"]]}
    assert get_column_signature(table) == "ncols_3"


def test_get_domain():
    table = {"pgId": 42}
    assert get_domain(table) == "42"


def test_get_domain_missing():
    table = {}
    assert get_domain(table) == "unknown"


def test_merge_tables_basic():
    tables = [
        {"relation": [["france", "germany"], ["fra", "deu"]]},
        {"relation": [["japan"], ["jpn"]]},
    ]
    pairs = merge_tables(tables)
    assert ("france", "fra") in pairs
    assert ("germany", "deu") in pairs
    assert ("japan", "jpn") in pairs


def test_merge_tables_deduplication():
    """Same pair from multiple tables should appear only once."""
    tables = [
        {"relation": [["france"], ["fra"]]},
        {"relation": [["france"], ["fra"]]},
    ]
    pairs = merge_tables(tables)
    assert pairs.count(("france", "fra")) == 1


def test_merge_tables_empty():
    assert merge_tables([]) == []


# ---------------------------------------------------------------------------
# Tests for UnionDomain
# ---------------------------------------------------------------------------

def test_union_domain_groups_by_page(sample_corpus):
    """Tables from same page with same structure should be merged."""
    results = union_domain(sample_corpus)
    assert len(results) > 0


def test_union_domain_separates_pages(sample_corpus):
    """Tables from different pages should be in different groups."""
    results = union_domain(sample_corpus)
    # Page 1 and Page 2 with same column count should be separate
    group_ids = [r["group_id"] for r in results]
    assert len(set(group_ids)) == len(group_ids)


def test_union_domain_method_label(sample_corpus):
    """All results should have method=UnionDomain."""
    results = union_domain(sample_corpus)
    for r in results:
        assert r["method"] == "UnionDomain"


def test_union_domain_empty():
    assert union_domain([]) == []


def test_union_domain_no_valid_tables():
    """Tables with less than 2 columns should be ignored."""
    corpus = [{"pgId": 1, "relation": [["france"]]}]
    assert union_domain(corpus) == []


# ---------------------------------------------------------------------------
# Tests for UnionWeb
# ---------------------------------------------------------------------------

def test_union_web_fewer_groups(sample_corpus):
    """UnionWeb should produce fewer groups than UnionDomain."""
    ud = union_domain(sample_corpus)
    uw = union_web(sample_corpus)
    assert len(uw) <= len(ud)


def test_union_web_method_label(sample_corpus):
    """All results should have method=UnionWeb."""
    results = union_web(sample_corpus)
    for r in results:
        assert r["method"] == "UnionWeb"


def test_union_web_more_pairs(sample_corpus):
    """UnionWeb should have more pairs per mapping than UnionDomain."""
    ud = union_domain(sample_corpus)
    uw = union_web(sample_corpus)
    if ud and uw:
        avg_ud = sum(r["size"] for r in ud) / len(ud)
        avg_uw = sum(r["size"] for r in uw) / len(uw)
        assert avg_uw >= avg_ud


def test_union_web_empty():
    assert union_web([]) == []


# ---------------------------------------------------------------------------
# Tests for SchemaCC
# ---------------------------------------------------------------------------

def test_schema_cc_returns_results(sample_corpus):
    """SchemaCC should return at least one mapping."""
    results = schema_cc(sample_corpus)
    assert len(results) > 0


def test_schema_cc_method_label(sample_corpus):
    """All results should have method=SchemaCC."""
    results = schema_cc(sample_corpus)
    for r in results:
        assert r["method"] == "SchemaCC"


def test_schema_cc_high_threshold(sample_corpus):
    """High threshold should produce more, smaller groups."""
    low = schema_cc(sample_corpus, threshold=0.1)
    high = schema_cc(sample_corpus, threshold=0.9)
    assert len(high) >= len(low)


def test_schema_cc_empty():
    assert schema_cc([]) == []


def test_schema_cc_no_valid_tables():
    corpus = [{"pgId": 1, "relation": [["france"]]}]
    assert schema_cc(corpus) == []