import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "output/wdc_run_100k/resolved_mappings.jsonl"
want = int(sys.argv[2]) if len(sys.argv) > 2 else 8

samples = []
with open(path) as f:
    for line in f:
        r = json.loads(line)
        pairs = r["pairs"]
        if 3 <= r["size"] <= 8 and all(len(str(a)) < 35 and len(str(b)) < 35 for a, b in pairs):
            samples.append(r)
        if len(samples) >= want:
            break

for r in samples:
    pid = r["partition_id"]
    sz = r["size"]
    cr = r["num_conflicts_removed"]
    print("--- partition %s | size=%s | conflicts_removed=%s ---" % (pid, sz, cr))
    for a, b in r["pairs"]:
        print("    %-38r <-> %r" % (a, b))
    print()
