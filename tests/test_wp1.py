"""Unit tests for WP1 (PMI coherence filtering).

Tests grow incrementally as each module is implemented per the plan."""

import json
import subprocess
import sys

from cooccurrence_index import (
    build_cooccurrence_index,
    index_summary,
    load_index,
    save_index,
)
from data_loader import (
    _load_csv_folder,
    _load_jsonl,
    clean_value,
    corpus_summary,
    load_corpus,
)
from filter import (
    filter_corpus,
    filtering_report,
    plot_coherence_distribution,
    rebuild_filtered_corpus,
    save_filtered_corpus,
    threshold_sweep,
)
from npmi import compute_pmi, compute_npmi, compute_coherence, score_corpus
from npmi import test_npmi as npmi_sanity_check


def test_clean_value_strips_and_lowercases():
    assert clean_value("  Germany  ") == "germany"


def test_clean_value_collapses_internal_whitespace():
    assert clean_value("New   York") == "new york"


def test_clean_value_handles_none():
    assert clean_value(None) == ""


def test_clean_value_handles_non_string():
    assert clean_value(42) == "42"
    assert clean_value(3.14) == "3.14"


def test_clean_value_handles_nan():
    import math
    assert clean_value(math.nan) == ""


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_jsonl_loader_reads_relation_as_columns(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["United States", "Canada"], ["USA", "CAN"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    assert len(corpus) == 1
    metadata, columns = corpus[0]
    assert columns == [["united states", "canada"], ["usa", "can"]]


def test_jsonl_loader_strips_header_row(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["Country", "USA", "Canada"], ["Code", "USA", "CAN"]],
        "tableType": "RELATION",
        "hasHeader": True,
        "headerRowIndex": 0,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    # Header row removed: "Country"/"Code" gone from each column
    assert columns == [["usa", "canada"], ["usa", "can"]]


def test_jsonl_loader_filters_table_types(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [
        {"relation": [["a", "b"], ["c", "d"]], "tableType": "LAYOUT",
         "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["a", "b"], ["c", "d"]], "tableType": "RELATION",
         "hasHeader": False, "headerRowIndex": -1},
    ])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    assert len(corpus) == 1


def test_jsonl_loader_drops_columns_below_2_unique(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["x", "x", "x"], ["a", "b", "c"]],  # first column has 1 unique
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    assert len(columns) == 1
    assert columns[0] == ["a", "b", "c"]


def test_jsonl_loader_drops_empty_columns(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["", "", ""], ["a", "b", "c"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    assert len(columns) == 1


def test_jsonl_loader_preserves_row_alignment(tmp_path):
    """After the WP1 row-alignment patch, columns of the same table
    keep equal length even when some cells are empty. The "" markers
    are preserved in place; PMI/coherence skip them downstream."""
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["a", "", "c"], ["x", "y", "z"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    # Both columns kept; both have length 3 (alignment preserved).
    assert len(columns) == 2
    assert len(columns[0]) == 3
    assert len(columns[1]) == 3
    assert columns[0] == ["a", "", "c"]
    assert columns[1] == ["x", "y", "z"]


def test_jsonl_loader_preserves_metadata(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["a", "b"], ["c", "d"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
        "pageTitle": "Hello",
        "url": "http://example.com",
        "tableNum": 7,
    }])
    corpus = _load_jsonl(str(p), table_types=("RELATION",), strip_headers=True)
    metadata, columns = corpus[0]
    assert metadata["pageTitle"] == "Hello"
    assert metadata["url"] == "http://example.com"
    assert metadata["tableNum"] == 7


def test_csv_folder_loads_and_transposes(tmp_path):
    csv_path = tmp_path / "table1.csv"
    csv_path.write_text("Country,Code\nUSA,USA\nCanada,CAN\nJapan,JPN\n", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    assert len(corpus) == 1
    metadata, columns = corpus[0]
    # CSV is read row-major then transposed; first row is treated as data,
    # which means "Country"/"Code" become regular values. CSV has no header
    # metadata so the loader cannot distinguish.
    assert columns[0] == ["country", "usa", "canada", "japan"]
    assert columns[1] == ["code", "usa", "can", "jpn"]


def test_csv_folder_loads_multiple_files(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("p,q\n5,6\n7,8\n", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    assert len(corpus) == 2


def test_csv_folder_skips_non_csv_files(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    assert len(corpus) == 1


def test_csv_folder_drops_short_columns(tmp_path):
    (tmp_path / "a.csv").write_text("a,b\nx,1\nx,2\n", encoding="utf-8")
    corpus = _load_csv_folder(str(tmp_path))
    metadata, columns = corpus[0]
    # First column ["a","x","x"] has 2 unique ("a","x"), kept.
    # Second column ["b","1","2"] has 3 unique, kept.
    assert len(columns) == 2


def test_load_corpus_dispatches_to_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{
        "relation": [["a", "b"], ["c", "d"]],
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
    }])
    corpus = load_corpus(str(p))
    assert len(corpus) == 1


def test_load_corpus_dispatches_to_csv_folder(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    corpus = load_corpus(str(tmp_path))
    assert len(corpus) == 1


def test_load_corpus_raises_on_missing_path(tmp_path):
    import pytest as _pt
    with _pt.raises(FileNotFoundError):
        load_corpus(str(tmp_path / "does_not_exist"))


def test_corpus_summary_runs(synthetic_corpus, capsys):
    corpus_summary(synthetic_corpus)
    captured = capsys.readouterr()
    assert "tables" in captured.out.lower()
    assert "columns" in captured.out.lower()


def test_index_total_columns(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # 5 country/ticker tables × 2 cols (10) + garbage (1) + 10 noise tables (10) = 21.
    assert N == 21


def test_index_value_count_dedupes_within_column(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # "united states" appears in column 0 of tables 0 and 1 only -> count 2.
    assert vc["united states"] == 2
    # "msft" appears in column 0 of tables 3 and 4 -> count 2.
    assert vc["msft"] == 2


def test_index_cooccurrence_uses_sorted_keys(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # ("canada", "united states") -> sorted key, present in tables 0 and 1.
    key = tuple(sorted(["united states", "canada"]))
    assert cooc[key] == 2
    # Reverse-order key should not exist.
    assert ("united states", "canada") not in cooc or key == ("canada", "united states")


def test_index_garbage_pairs_unique_to_one_column(synthetic_corpus):
    cooc, vc, N = build_cooccurrence_index(synthetic_corpus)
    # ("hello world", "the matrix") only co-occurs in the garbage column.
    key = tuple(sorted(["hello world", "the matrix"]))
    assert cooc[key] == 1


def test_index_pickle_roundtrip(synthetic_corpus, tmp_path):
    original = build_cooccurrence_index(synthetic_corpus)
    p = tmp_path / "idx.pkl"
    save_index(original, str(p))
    loaded = load_index(str(p))
    assert loaded[0] == original[0]
    assert loaded[1] == original[1]
    assert loaded[2] == original[2]


def test_index_summary_runs(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    index_summary(idx)
    captured = capsys.readouterr()
    assert "Top" in captured.out or "top" in captured.out


def test_npmi_perfect_co_occurrence(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    # "united states" and "canada" are both in column 0 of tables 0 and 1,
    # and they only appear there. p(uv) = p(u) = p(v) = 2/21 -> NPMI = +1.
    score = compute_npmi("united states", "canada", idx)
    assert score == 1.0


def test_npmi_never_co_occur(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    # "united states" never appears with "msft".
    score = compute_npmi("united states", "msft", idx)
    assert score == -1.0


def test_npmi_unknown_value_returns_minus_one(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    score = compute_npmi("united states", "this_value_is_not_in_corpus", idx)
    assert score == -1.0


def test_npmi_symmetry(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    cooc, vc, N = idx
    pairs = list(cooc.keys())[:50]
    for u, v in pairs:
        assert compute_npmi(u, v, idx) == compute_npmi(v, u, idx)


def test_npmi_clipping_range(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    cooc, vc, N = idx
    for u, v in list(cooc.keys())[:200]:
        s = compute_npmi(u, v, idx)
        assert -1.0 <= s <= 1.0


def test_pmi_returns_negative_inf_when_pair_missing(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    assert compute_pmi("united states", "msft", idx) == float("-inf")


def test_coherence_coherent_column_high(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    # First country column of table 0.
    s = compute_coherence(["united states", "canada", "japan"], idx)
    assert s > 0.5


def test_coherence_garbage_column_low(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    s = compute_coherence(
        ["2024-01-01", "hello world", "83.5%", "the matrix", "blue"], idx
    )
    # All pairs never co-occur outside this single column, so NPMI for each
    # pair is < +1 (specifically, computed value), but the column is the
    # only place these values appear together. Coherence should still be
    # *lower* than coherent columns and clearly < 0.3 threshold.
    assert s < 0.3


def test_coherence_coherent_outranks_garbage(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    coherent = compute_coherence(["united states", "canada", "japan"], idx)
    garbage = compute_coherence(
        ["2024-01-01", "hello world", "83.5%", "the matrix", "blue"], idx
    )
    assert coherent > garbage


def test_score_corpus_returns_one_per_column(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    total_cols = sum(len(cols) for _, cols in synthetic_corpus)
    assert len(scored) == total_cols


def test_score_corpus_tuple_shape(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    table_idx, col_idx, values, score = scored[0]
    assert isinstance(table_idx, int)
    assert isinstance(col_idx, int)
    assert isinstance(values, list)
    assert isinstance(score, float)


def test_score_corpus_indices_correct(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    # Find the entry for table 5 (garbage). It has only one column.
    table5 = [s for s in scored if s[0] == 5]
    assert len(table5) == 1
    assert table5[0][1] == 0


def test_runtime_sanity_helper_prints(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    npmi_sanity_check(idx)
    captured = capsys.readouterr()
    assert "NPMI" in captured.out or "npmi" in captured.out


def test_filter_partitions_above_and_below_threshold(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    assert all(s >= 0.3 for _, _, _, s in kept)
    assert all(s < 0.3 for _, _, _, s in removed)
    assert len(kept) + len(removed) == len(scored)


def test_filter_garbage_column_removed(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    # Table 5 column 0 is the garbage column.
    removed_ids = {(ti, ci) for ti, ci, _, _ in removed}
    assert (5, 0) in removed_ids


def test_filter_country_columns_kept(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    kept_ids = {(ti, ci) for ti, ci, _, _ in kept}
    # Tables 0-2 are country/iso, both columns should survive.
    for ti in (0, 1, 2):
        assert (ti, 0) in kept_ids
        assert (ti, 1) in kept_ids


def test_rebuild_keeps_only_surviving_columns(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    # Each filtered entry: (metadata, kept_columns, kept_scores, rejected_indices)
    for metadata, columns, scores, rejected in filtered:
        assert len(columns) == len(scores)
        assert all(isinstance(s, float) for s in scores)


def test_rebuild_drops_tables_with_zero_kept(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    # Table 5 (garbage column) has zero kept columns -> dropped.
    # Original corpus has 16 tables; only 1 (table 5) is fully dropped.
    assert len(filtered) == 15


def test_rebuild_records_rejected_indices(synthetic_corpus):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    # In synthetic_corpus tables 0-4 and 6-15, all columns survive,
    # so rejected_indices is empty for those tables.
    for metadata, columns, scores, rejected in filtered:
        assert rejected == []


def test_save_filtered_corpus_jsonl_schema(synthetic_corpus, tmp_path):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(synthetic_corpus, kept)
    out = tmp_path / "filtered.jsonl"
    save_filtered_corpus(filtered, str(out))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    for line in lines:
        rec = json.loads(line)
        assert "relation" in rec
        assert "coherence_scores" in rec
        assert "rejected_column_indices" in rec
        assert len(rec["relation"]) == len(rec["coherence_scores"])


def test_save_filtered_corpus_preserves_metadata(tmp_path):
    # Tiny corpus with metadata attached so we can verify it carries through.
    corpus = [
        ({"pageTitle": "X", "url": "http://x"},
         [["united states", "canada"], ["usa", "can"]]),
    ]
    # Build a small index that makes the columns coherent.
    idx_corpus = corpus + [
        ({}, [["united states", "canada"], ["usa", "can"]]),
        ({}, [["united states", "canada"], ["usa", "can"]]),
    ]
    idx = build_cooccurrence_index(idx_corpus)
    scored = score_corpus(corpus, idx)
    kept, _ = filter_corpus(scored, threshold=0.3)
    filtered = rebuild_filtered_corpus(corpus, kept)
    out = tmp_path / "filtered.jsonl"
    save_filtered_corpus(filtered, str(out))
    rec = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["pageTitle"] == "X"
    assert rec["url"] == "http://x"


def test_filtering_report_runs(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    kept, removed = filter_corpus(scored, threshold=0.3)
    filtering_report(kept, removed)
    out = capsys.readouterr().out
    assert "before" in out.lower()
    assert "after" in out.lower()


def test_threshold_sweep_runs(synthetic_corpus, capsys):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    threshold_sweep(scored, thresholds=(0.1, 0.3, 0.5))
    out = capsys.readouterr().out
    assert "0.1" in out
    assert "0.3" in out
    assert "0.5" in out


def test_plot_coherence_distribution_writes_file(synthetic_corpus, tmp_path):
    idx = build_cooccurrence_index(synthetic_corpus)
    scored = score_corpus(synthetic_corpus, idx)
    out = tmp_path / "hist.png"
    plot_coherence_distribution(scored, threshold=0.3, output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_main_end_to_end_on_synthetic_jsonl(tmp_path):
    """Smoke test: run main.py against a tiny synthetic JSONL and verify
    all four output artifacts (WP1 + WP2) exist."""
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        {"relation": [["united states", "canada", "japan"], ["usa", "can", "jpn"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["united states", "canada", "germany"], ["usa", "can", "deu"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["japan", "germany", "france"], ["jpn", "deu", "fra"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["msft", "aapl"], ["microsoft", "apple"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["msft", "aapl", "googl"], ["microsoft", "apple", "alphabet"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
        {"relation": [["2024-01-01", "hello world", "83.5%", "the matrix", "blue"]],
         "tableType": "RELATION", "hasHeader": False, "headerRowIndex": -1},
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    idx = tmp_path / "idx.pkl"
    result = subprocess.run(
        [sys.executable, "main.py",
         "--corpus_path", str(corpus_path),
         "--output_folder", str(out),
         "--threshold", "0.3",
         "--index_path", str(idx)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "filtered_corpus.jsonl").exists()
    assert (out / "coherence_distribution.png").exists()
    assert (out / "threshold_sweep.txt").exists()
    assert (out / "candidates.jsonl").exists()
    assert idx.exists()
