"""pytest rootdir marker. Putting this file at the repo root makes pytest
add the root to sys.path so tests can `import data_loader`, `import npmi`,
etc. without a package layout."""

import pytest


@pytest.fixture
def synthetic_corpus():
    """Synthetic mini-corpus for unit tests.

    Five coherent table pairs (countries / ISO codes, tickers / companies),
    one incoherent garbage column, and ten "noise" tables that place each
    garbage value in two unrelated coherent contexts. The noise is what
    makes NPMI distinguish the garbage column from the coherent ones —
    without it, garbage pairs would score NPMI=+1 because each value would
    only appear in the one garbage column.

    21 columns total. Designed so coherent country/ticker columns score
    near +0.7 and the garbage column scores near +0.28.
    """
    return [
        # Coherent: countries / ISO codes
        ({}, [["united states", "canada", "japan"], ["usa", "can", "jpn"]]),       # table 0
        ({}, [["united states", "canada", "germany"], ["usa", "can", "deu"]]),     # table 1
        ({}, [["japan", "germany", "france"], ["jpn", "deu", "fra"]]),             # table 2
        # Coherent: tickers / companies
        ({}, [["msft", "aapl", "googl"], ["microsoft", "apple", "alphabet"]]),     # table 3
        ({}, [["msft", "aapl"], ["microsoft", "apple"]]),                          # table 4
        # Incoherent: a column mixing values from unrelated domains
        ({}, [["2024-01-01", "hello world", "83.5%", "the matrix", "blue"]]),      # table 5
        # Noise: dates  (puts "2024-01-01" in a real date column)
        ({}, [["2024-01-01", "2024-06-15", "2024-12-31"]]),                        # table 6
        ({}, [["2024-01-01", "2024-02-29"]]),                                      # table 7
        # Noise: colors  (puts "blue" in a real color column)
        ({}, [["red", "blue", "green"]]),                                          # table 8
        ({}, [["yellow", "blue", "purple"]]),                                      # table 9
        # Noise: movies  (puts "the matrix" in a real movie column)
        ({}, [["the matrix", "inception"]]),                                       # table 10
        ({}, [["the matrix", "interstellar"]]),                                    # table 11
        # Noise: percentages  (puts "83.5%" in a real percentage column)
        ({}, [["83.5%", "67.1%", "12.5%"]]),                                       # table 12
        ({}, [["83.5%", "45.2%"]]),                                                # table 13
        # Noise: greetings  (puts "hello world" in a real greeting column)
        ({}, [["hello world", "goodbye"]]),                                        # table 14
        ({}, [["hello world", "hi there"]]),                                       # table 15
    ]


@pytest.fixture
def fd_synthetic_table():
    """Five row-aligned columns, 8 rows. Designed for FD test cases.

    Layout (one row per index, one column per list):
        col 0  LEFT_CC      col 1  RIGHT_CC    col 2  AMBIG
        col 3  FOREIGN      col 4  CONST

    Row 0:  united     usa       portland   yes  A
    Row 1:  united     usa       portland   yes  A
    Row 2:  canada     can       vancouver  no   A
    Row 3:  japan      jpn       tokyo      yes  A
    Row 4:  germany    deu       berlin     yes  A
    Row 5:  france     fra       paris      no   A
    Row 6:  portland   oregon    ""         yes  A     <- "portland" reused, ambiguous
    Row 7:  portland   maine     ""         no   A

    Designed so that:
      - (LEFT_CC, AMBIG) after empty-row filtering has 6 perfect-FD rows -> theta = 1.0
      - (LEFT_CC, RIGHT_CC) over the full 8 rows has theta = 7/8 = 0.875
        (the witness subset keeps both 'united'->'usa' rows, drops
        one of the ambiguous 'portland' rows)
      - (LEFT_CC, CONST) has only 1 distinct Y -> rejected
    """
    return ({}, [
        ["united", "united", "canada", "japan", "germany", "france", "portland", "portland"],
        ["usa",    "usa",    "can",    "jpn",   "deu",     "fra",    "oregon",   "maine"],
        ["portland","portland","vancouver","tokyo","berlin","paris","",          ""],
        ["yes",    "yes",    "no",     "yes",   "yes",     "no",     "yes",      "no"],
        ["A",      "A",      "A",      "A",     "A",       "A",      "A",        "A"],
    ])


@pytest.fixture
def synthesis_candidates():
    """Hand-crafted 8 candidates for synthesis tests.

    Designed to exercise:
      - perfect merge (0 ↔ 3, identical pairs)
      - partial overlap (0 ↔ 1, two shared pairs)
      - conflict / negative score (0 ↔ 2, same lefts, different rights)
      - isolated singletons (4, 7 — share nothing)
      - identity / degenerate (5 — left == right per pair)
      - rank-numeric (6 — numeric left, won't merge)
      - no-overlap baseline (0 ↔ 6 share zero pairs despite both having "1" in them)

    Each candidate is a dict matching what `load_candidates` returns:
    `{"pairs": [(l, r), ...], "theta": float, "row_count": int,
      "covered_rows": int, "source_table_index": int,
      "left_column_index": int, "right_column_index": int,
      "source_metadata": {}}`.
    """
    def mk(idx, pairs):
        return {
            "pairs": [tuple(p) for p in pairs],
            "theta": 1.0,
            "row_count": len(pairs),
            "covered_rows": len(pairs),
            "source_table_index": idx,
            "left_column_index": 0,
            "right_column_index": 1,
            "source_metadata": {},
        }
    return [
        mk(0, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell")]),
        mk(1, [("dune", "herbert"), ("ender", "card"), ("foundation", "asimov")]),
        mk(2, [("dune", "1965"), ("foundation", "1951"), ("1984", "1949")]),
        mk(3, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell")]),
        mk(4, [("france", "paris"), ("japan", "tokyo"), ("egypt", "cairo")]),
        mk(5, [("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma")]),
        mk(6, [("1", "100"), ("2", "200"), ("3", "300")]),
        mk(7, [("red", "warm"), ("blue", "cool"), ("green", "neutral")]),
    ]


@pytest.fixture
def synthesis_components_fixture():
    """7 candidates for connected-component tests.

    Designed structure:
      - 0, 1, 2 form positive-overlap cluster A ("book → author")
      - 3, 4, 5 form positive-overlap cluster B ("country → capital")
      - 6 is isolated (shares nothing with anyone)

    With theta_edge=0 and theta_overlap=1 the positive-edge graph has
    three connected components: {0,1,2}, {3,4,5}, {6}.
    """
    def mk(idx, pairs):
        return {
            "pairs": [tuple(p) for p in pairs],
            "theta": 1.0,
            "row_count": len(pairs),
            "covered_rows": len(pairs),
            "source_table_index": idx,
            "left_column_index": 0,
            "right_column_index": 1,
            "source_metadata": {},
        }
    return [
        # Cluster A — book → author
        mk(0, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell")]),
        mk(1, [("dune", "herbert"), ("ender", "card"), ("foundation", "asimov")]),
        mk(2, [("dune", "herbert"), ("foundation", "asimov"), ("1984", "orwell"), ("hobbit", "tolkien")]),
        # Cluster B — country → capital
        mk(3, [("france", "paris"), ("germany", "berlin"), ("italy", "rome")]),
        mk(4, [("france", "paris"), ("spain", "madrid"), ("germany", "berlin")]),
        mk(5, [("france", "paris"), ("germany", "berlin"), ("italy", "rome"), ("uk", "london")]),
        # Isolated singleton — no pair-overlap with anyone
        mk(6, [("red", "warm"), ("blue", "cool"), ("green", "neutral")]),
    ]
