"""Scan the full WDC corpus for tables that contain benchmark (gold) pairs.

The pipeline can only produce a value-correspondence pair (a, b) if a and b
CO-OCCUR in a row of some RELATION table (that's what seeds the candidate). So
to give the pipeline a chance at every benchmark mapping, the sample must
include the tables where each gold pair co-occurs.

This streams the nested WDC archives, applies the same filtering/normalization
as data_loader (RELATION only, header stripped, clean_value on cells), and:
  - writes every table that contains >=1 gold pair to an output JSONL (the
    coverage sample), and
  - reports, for each gold pair, whether it is COVERED (co-occurs in a row),
    PRESENT-BUT-SPLIT (both values seen somewhere but never in the same row),
    or ABSENT (>=1 value never seen at all).

Usage:
    python build_benchmark_cover.py <benchmark.txt> <out.jsonl> <archive1> [archive2 ...]
"""

import json
import sys
import tarfile

from data_loader import clean_value

BENCH = sys.argv[1]
OUT = sys.argv[2]
ARCHIVES = sys.argv[3:]


def log(msg):
    print(msg, flush=True)


# ---- gold pairs ----
G = set()                       # frozenset({a, b})
partners = {}                   # value -> set of partner values
V_gold = set()
with open(BENCH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        a, b = clean_value(p[1]), clean_value(p[2])
        if not a or not b or a == b:
            continue
        G.add(frozenset((a, b)))
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
        V_gold.add(a)
        V_gold.add(b)
log("Gold: %d distinct pairs, %d distinct values" % (len(G), len(V_gold)))

covered = set()          # gold pairs found co-occurring in a row
seen_gold_vals = set()   # gold values seen anywhere in corpus
n_tables = 0
n_kept = 0


def iter_records(archive):
    outer = tarfile.open(archive, "r:gz")
    try:
        for om in outer:
            if not om.isfile():
                continue
            fo = outer.extractfile(om)
            if fo is None:
                continue
            inner = tarfile.open(fileobj=fo, mode="r|")
            for tm in inner:
                if not tm.name.endswith(".json"):
                    continue
                f = inner.extractfile(tm)
                if f is None:
                    continue
                try:
                    yield json.loads(f.read())
                except Exception:
                    continue
            inner.close()
            break
    finally:
        outer.close()


out = open(OUT, "w", encoding="utf-8")
for archive in ARCHIVES:
    log("=== scanning %s ===" % archive)
    for rec in iter_records(archive):
        n_tables += 1
        if n_tables % 100000 == 0:
            log("  ...%d tables scanned, %d kept, %d/%d gold pairs covered"
                % (n_tables, n_kept, len(covered), len(G)))
        if rec.get("tableType") != "RELATION":
            continue
        relation = rec.get("relation") or []
        if not relation:
            continue
        hri = rec["headerRowIndex"] if (rec.get("hasHeader") and rec.get("headerRowIndex", -1) >= 0) else -1
        nrows = max((len(c) for c in relation), default=0)
        table_hits = set()
        for r in range(nrows):
            if r == hri:
                continue
            row_gold = []
            for col in relation:
                if r < len(col):
                    v = clean_value(col[r])
                    if v in V_gold:
                        row_gold.append(v)
                        seen_gold_vals.add(v)
            if len(row_gold) < 2:
                continue
            rg = set(row_gold)
            for v in rg:
                for u in partners.get(v, ()):  # only real gold partners
                    if u in rg:
                        table_hits.add(frozenset((v, u)))
        if table_hits:
            covered |= table_hits
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
out.close()

# ---- classify uncovered pairs ----
present_but_split = 0
absent = 0
for pair in G:
    if pair in covered:
        continue
    if all(v in seen_gold_vals for v in pair):
        present_but_split += 1
    else:
        absent += 1

log("")
log("=== COVERAGE RESULT ===")
log("Tables scanned:            %d" % n_tables)
log("Tables kept (cover sample):%d" % n_kept)
log("Gold pairs total:          %d" % len(G))
log("  COVERED (co-occur):      %d (%.1f%%)" % (len(covered), 100.0 * len(covered) / len(G)))
log("  PRESENT-BUT-SPLIT:       %d (%.1f%%)" % (present_but_split, 100.0 * present_but_split / len(G)))
log("  ABSENT (>=1 val missing):%d (%.1f%%)" % (absent, 100.0 * absent / len(G)))
log("Cover sample written to:   %s" % OUT)
