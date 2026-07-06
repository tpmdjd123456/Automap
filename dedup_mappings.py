import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "output/wdc_run_100k/resolved_mappings.jsonl"

total = 0
total_pairs = 0
# direction-sensitive: dedup byte-identical mappings (ordered pairs, as a set)
dir_sensitive = set()
# direction-insensitive: treat (a,b) == (b,a); mapping = set of unordered pairs
dir_insensitive = set()

with open(path) as f:
    for line in f:
        r = json.loads(line)
        pairs = r["pairs"]
        total += 1
        total_pairs += len(pairs)

        ordered = frozenset((str(a), str(b)) for a, b in pairs)
        dir_sensitive.add(ordered)

        unordered = frozenset(
            frozenset((str(a), str(b))) for a, b in pairs
        )
        dir_insensitive.add(unordered)

print("Raw partitions (mappings):        %d" % total)
print("Total value-pairs:                %d" % total_pairs)
print("Distinct (dedup identical sets):  %d" % len(dir_sensitive))
print("Distinct (direction-insensitive): %d" % len(dir_insensitive))
print()
print("Identical-set duplicates removed: %d (%.1f%%)" % (
    total - len(dir_sensitive), 100.0 * (total - len(dir_sensitive)) / total))
print("After also merging symmetric:     %d distinct (%.1f%% of raw)" % (
    len(dir_insensitive), 100.0 * len(dir_insensitive) / total))
