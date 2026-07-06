"""Build an N-table corpus sample that CONTAINS every cover table (so every
covered benchmark pair co-occurs in the sample and the full recall ceiling is
reachable in-sample), while staying close in size and composition to an existing
baseline sample.

    new sample = all cover tables
               + non-cover tables from the baseline sample, truncated to N total

Injecting the cover tables into the baseline and evicting an equal number of
non-cover tables keeps the sample maximally comparable to the baseline run: the
only difference is the tables that were actually needed for coverage.

Tables are de-duplicated by an MD5 of their canonical JSON so a cover table
already present in the baseline is not written twice.

Usage:
    python build_covered_sample.py <cover.jsonl> <baseline.jsonl> <N> <out.jsonl>
"""

import hashlib
import json
import sys

COVER = sys.argv[1]
BASE = sys.argv[2]
N = int(sys.argv[3])
OUT = sys.argv[4]


def log(msg):
    print(msg, flush=True)


def key(rec):
    return hashlib.md5(
        json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def wln(fh, line):
    fh.write(line if line.endswith("\n") else line + "\n")


out = open(OUT, "w", encoding="utf-8")

# ---- 1) write every cover table, remember their identities ----
cover_keys = set()
n_cover = 0
with open(COVER, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        cover_keys.add(key(rec))
        wln(out, line)
        n_cover += 1
log("Cover tables written: %d (distinct: %d)" % (n_cover, len(cover_keys)))

need = N - n_cover
if need < 0:
    log("WARNING: cover set (%d) already exceeds target N (%d); no filler added."
        % (n_cover, N))
    need = 0

# ---- 2) fill from the baseline with non-cover tables until we hit N ----
added = 0
overlap = 0          # baseline tables that ARE cover tables (already included)
n_base = 0
with open(BASE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not line.strip():
            continue
        n_base += 1
        rec = json.loads(line)
        if key(rec) in cover_keys:
            overlap += 1
            continue
        if added < need:
            wln(out, line)
            added += 1
out.close()

total = n_cover + added
log("Baseline tables scanned:  %d" % n_base)
log("  already-cover (overlap):%d (%.1f%% of baseline)"
    % (overlap, 100.0 * overlap / n_base if n_base else 0.0))
log("Filler tables added:      %d (target %d)" % (added, need))
log("")
log("=== SAMPLE BUILT ===")
log("Total tables written:     %d (target %d)" % (total, N))
log("  cover tables:           %d" % n_cover)
log("  baseline filler:        %d" % added)
if total < N:
    log("WARNING: only %d tables (baseline ran out of non-cover tables)." % total)
log("Written to:               %s" % OUT)
