"""Build an N-table WEB-TABLE sample that CONTAINS every cover table, streaming
filler straight from the raw WDC archives (no pre-built baseline jsonl needed).

    new sample = all cover tables (guarantees every covered pair co-occurs)
               + raw archive records (in archive order, minus cover dups)
                 until N tables total

Because every cover table is included, the reachable/achievable benchmark within
this sample is the full covered set, so recall can be scored against it.

Usage:
    python build_covered_sample_stream.py <cover.jsonl> <N> <out.jsonl> <archive1> [archive2 ...]
"""

import hashlib
import json
import sys
import tarfile

COVER = sys.argv[1]
N = int(sys.argv[2])
OUT = sys.argv[3]
ARCHIVES = sys.argv[4:]


def log(msg):
    print(msg, flush=True)


def key(rec):
    return hashlib.md5(
        json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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
            break  # single inner tar per archive
    finally:
        outer.close()


out = open(OUT, "w", encoding="utf-8")

# ---- 1) write every cover table, remember identities ----
cover_keys = set()
n_cover = 0
with open(COVER, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        cover_keys.add(key(rec))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_cover += 1
log("Cover tables written: %d" % n_cover)

need = N - n_cover
if need < 0:
    need = 0
    log("WARNING: cover set (%d) exceeds target N (%d); no filler." % (n_cover, N))

# ---- 2) stream filler from archives until we reach N ----
added = 0
scanned = 0
dup = 0
for archive in ARCHIVES:
    if added >= need:
        break
    log("=== streaming filler from %s ===" % archive)
    for rec in iter_records(archive):
        if added >= need:
            break
        scanned += 1
        if key(rec) in cover_keys:
            dup += 1
            continue
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        added += 1
        if added % 100000 == 0:
            log("  ...%d filler written (%d scanned, %d cover-dups skipped)"
                % (added, scanned, dup))
out.close()

total = n_cover + added
log("")
log("=== SAMPLE BUILT ===")
log("Cover tables:        %d" % n_cover)
log("Filler tables:       %d (scanned %d, skipped %d cover dups)" % (added, scanned, dup))
log("Total tables:        %d (target %d)" % (total, N))
if total < N:
    log("WARNING: archives exhausted before reaching N (%d < %d)." % (total, N))
log("Written to:          %s" % OUT)
