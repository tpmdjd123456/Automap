"""Write the COVERED subset of the benchmark: gold pairs that actually co-occur
in a row of some RELATION table in the corpus.

This reconstructs the exact `covered` set from build_benchmark_cover.py by
re-applying the same row co-occurrence logic to the cover sample it emitted
(every kept table has >=1 gold co-occurrence, so the kept tables reproduce the
full covered set). Reading the ~47k-table cover sample avoids rescanning the 2M
table archives.

The pipeline can only ever predict a covered pair, so covered recall
(|P & covered| / |covered|) is the true achievable-recall ceiling.

Usage:
    python extract_covered_pairs.py <benchmark.txt> <cover.jsonl> <out_covered.txt>
"""

import json
import sys

from data_loader import clean_value

BENCH = sys.argv[1]
COVER = sys.argv[2]
OUT = sys.argv[3]


def log(msg):
    print(msg, flush=True)


# ---- gold pairs (same normalization as build_benchmark_cover) ----
partners = {}          # value -> set of partner values
V_gold = set()
with open(BENCH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        a, b = clean_value(p[1]), clean_value(p[2])
        if not a or not b or a == b:
            continue
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
        V_gold.add(a)
        V_gold.add(b)

# ---- reconstruct covered pairs from the cover sample ----
covered = set()
n_tables = 0
with open(COVER, encoding="utf-8", errors="ignore") as f:
    for line in f:
        rec = json.loads(line)
        n_tables += 1
        relation = rec.get("relation") or []
        if not relation:
            continue
        hri = rec["headerRowIndex"] if (rec.get("hasHeader") and rec.get("headerRowIndex", -1) >= 0) else -1
        nrows = max((len(c) for c in relation), default=0)
        for r in range(nrows):
            if r == hri:
                continue
            rg = set()
            for col in relation:
                if r < len(col):
                    v = clean_value(col[r])
                    if v in V_gold:
                        rg.add(v)
            if len(rg) < 2:
                continue
            for v in rg:
                for u in partners.get(v, ()):
                    if u in rg:
                        covered.add(frozenset((v, u)))
log("Cover tables read: %d" % n_tables)
log("Covered gold pairs: %d" % len(covered))

# ---- write covered subset, preserving original benchmark line format ----
n_written = 0
seen = set()
with open(BENCH, encoding="utf-8", errors="ignore") as f, \
        open(OUT, "w", encoding="utf-8") as g:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        a, b = clean_value(p[1]), clean_value(p[2])
        if not a or not b or a == b:
            continue
        key = frozenset((a, b))
        if key in covered and key not in seen:
            seen.add(key)
            g.write(line if line.endswith("\n") else line + "\n")
            n_written += 1
log("Lines written to %s: %d" % (OUT, n_written))
