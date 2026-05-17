"""Unit tests for WP2 (column-pair filtering by approximate FD).

Tests grow incrementally per the implementation plan."""

from fd_filter import compute_approx_fd



def test_perfect_fd_passes(fd_synthetic_table):
    """LEFT_CC -> AMBIG has 2 empty rows (6, 7) which are dropped, leaving
    6 rows where each distinct LEFT value maps to exactly one AMBIG value.
    theta should be 1.0."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[2]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert theta == 1.0
    assert row_count == 6
    assert covered == 6


def test_name_ambiguity_rejected_at_095(fd_synthetic_table):
    """LEFT_CC -> RIGHT_CC over the full 8 rows.

    Witness subset: 'united' appears twice with 'usa' (both rows kept,
    contributing 2). canada/japan/germany/france each contribute 1.
    'portland' appears in rows 6,7 with two different y values; pick
    one most-common-y, the other row is excluded — contributes 1.
    Covered = 2+1+1+1+1+1 = 7. theta = 7/8 = 0.875 < 0.95."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[1]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert row_count == 8
    assert covered == 7
    assert theta == 0.875


def test_name_ambiguity_accepted_at_lower_theta(fd_synthetic_table):
    """The ambiguous case has theta = 0.875 (see test above for math).
    With theta_threshold = 0.85, the higher-level filter would accept
    it; with 0.95 it rejects. compute_approx_fd itself doesn't apply
    the threshold — it just computes theta."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[1]
    theta, _, _, _ = compute_approx_fd(left, right)
    assert theta == 0.875


def test_constant_column_rejected(fd_synthetic_table):
    """LEFT_CC -> CONST: Y has only 1 distinct value ('A'). Reject."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[4]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert theta == 0.0
    assert pairs == []
    assert row_count == 0


def test_too_few_rows_rejected():
    """A pair with fewer than min_rows non-empty rows is rejected."""
    left = ["a", "b", ""]
    right = ["x", "y", ""]
    theta, pairs, row_count, covered = compute_approx_fd(left, right, min_rows=3)
    assert theta == 0.0


def test_empty_rows_dropped(fd_synthetic_table):
    """LEFT_CC -> AMBIG: rows 6 and 7 have AMBIG=''; they're dropped before
    FD computation. Remaining 6 rows give a perfect FD."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[2]
    theta, pairs, row_count, covered = compute_approx_fd(left, right)
    assert row_count == 6  # 8 total - 2 empty AMBIG rows


def test_pairs_are_deduplicated(fd_synthetic_table):
    """LEFT_CC -> AMBIG: 'united' appears twice mapping to 'portland' both
    times. The output 'pairs' list should contain ('united', 'portland')
    exactly once."""
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[2]
    _, pairs, _, _ = compute_approx_fd(left, right)
    united_entries = [p for p in pairs if p[0] == "united"]
    assert len(united_entries) == 1
    assert united_entries[0] == ("united", "portland")


def test_surviving_pairs_match_witness_subset(fd_synthetic_table):
    """For each x in pairs, the y is the most common y in the original
    column among rows where left == x and both values are non-empty."""
    from collections import Counter
    _, cols = fd_synthetic_table
    left, right = cols[0], cols[1]  # has ambiguity
    _, pairs, _, _ = compute_approx_fd(left, right)
    # Build the ground-truth most-common-y for each x.
    rows = [(x, y) for x, y in zip(left, right) if x and y]
    by_x = {}
    for x, y in rows:
        by_x.setdefault(x, Counter())[y] += 1
    expected = {x: counter.most_common(1)[0][0] for x, counter in by_x.items()}
    actual = dict(pairs)
    assert actual == expected


from fd_filter import filter_candidates_by_fd


def test_ordered_pairs_are_distinct_candidates(fd_synthetic_table):
    """A 2-column table evaluated with FD yields up to 2 candidates:
    (left=0, right=1) and (left=1, right=0). They're independent and
    can have different theta values."""
    _, cols = fd_synthetic_table
    # Use just LEFT_CC and AMBIG as a 2-column table (perfect FD both
    # ways after empty-row filtering on either side).
    record = {
        "relation": [cols[0], cols[2]],
        "coherence_scores": [1.0, 1.0],
        "rejected_column_indices": [],
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    indices = {(c["left_column_index"], c["right_column_index"]) for c in candidates}
    assert (0, 1) in indices
    assert (1, 0) in indices


def test_filter_candidates_passes_through_metadata():
    """source_metadata should pass through input record fields except
    relation, coherence_scores, and rejected_column_indices."""
    record = {
        "relation": [
            ["a", "b", "c", "d"],
            ["1", "2", "3", "4"],
        ],
        "coherence_scores": [1.0, 1.0],
        "rejected_column_indices": [2, 5],
        "pageTitle": "Hello",
        "url": "http://example.com",
        "tableType": "RELATION",
        "tableNum": 7,
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    assert len(candidates) >= 1
    meta = candidates[0]["source_metadata"]
    assert meta["pageTitle"] == "Hello"
    assert meta["url"] == "http://example.com"
    assert meta["tableType"] == "RELATION"
    assert meta["tableNum"] == 7
    assert "relation" not in meta
    assert "coherence_scores" not in meta
    assert "rejected_column_indices" not in meta


def test_filter_candidates_includes_full_schema():
    """Every produced candidate has the 8 spec'd fields with correct types."""
    record = {
        "relation": [
            ["a", "b", "c", "d"],
            ["1", "2", "3", "4"],
        ],
        "pageTitle": "X",
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    assert len(candidates) >= 1
    c = candidates[0]
    assert isinstance(c["pairs"], list)
    assert isinstance(c["theta"], float)
    assert isinstance(c["row_count"], int)
    assert isinstance(c["covered_rows"], int)
    assert isinstance(c["source_table_index"], int)
    assert isinstance(c["left_column_index"], int)
    assert isinstance(c["right_column_index"], int)
    assert isinstance(c["source_metadata"], dict)


import json
from fd_filter import save_candidates


def test_save_candidates_jsonl_schema(fd_synthetic_table, tmp_path):
    """Round-trip: save_candidates writes valid JSONL where each line
    parses back to the same Candidate dict."""
    _, cols = fd_synthetic_table
    record = {
        "relation": [cols[0], cols[2]],
        "pageTitle": "Test",
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    out = tmp_path / "candidates.jsonl"
    save_candidates(candidates, str(out))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(candidates)
    for line, original in zip(lines, candidates):
        parsed = json.loads(line)
        # 8 required fields
        for field in ("pairs", "theta", "row_count", "covered_rows",
                      "source_table_index", "left_column_index",
                      "right_column_index", "source_metadata"):
            assert field in parsed
        # JSON converts tuples to lists; original `pairs` was already lists.
        assert parsed["theta"] == original["theta"]
        assert parsed["pairs"] == original["pairs"]


from fd_filter import candidates_summary


def test_candidates_summary_runs(fd_synthetic_table, capsys):
    _, cols = fd_synthetic_table
    record = {
        "relation": [cols[0], cols[2]],
        "pageTitle": "Test",
    }
    candidates = filter_candidates_by_fd([record], theta_threshold=0.95)
    candidates_summary(candidates)
    out = capsys.readouterr().out
    assert "Candidates" in out
    assert "Theta" in out or "theta" in out


def test_candidates_summary_handles_empty_input(capsys):
    candidates_summary([])
    out = capsys.readouterr().out
    assert "No candidates" in out or "0 candidates" in out.lower()
