# Sample outputs — 1.5M-table Jaccard run

The full run lives **outside git** because it is ~18 GB. This directory holds a
small, committable sample so the output format can be inspected without
transferring the real data.

| File | What it is |
|------|------------|
| `resolved_mappings.sample.jsonl` | First 1,000 records of the final output (`resolved_mappings.jsonl`). |
| `threshold_sweep.txt` | Full conflict-resolution threshold sweep (already tiny). |
| `coherence_distribution.png` | Partition-coherence distribution plot. |

## Full results location (not in git)

- **Local:** `output/results_1500k_jaccard/`
- **dama:** `/home/automap/Automap/output/results_1500k_jaccard/`

Final output: `resolved_mappings.jsonl` — **10,734,449 records**, 4.0 GB.
The whole folder (candidates, edge scores, synthesized + resolved mappings,
filtered corpus) is 18 GB.

## Record schema (`resolved_mappings.jsonl`)

One JSON object per line. Each record is a resolved partition of attribute
name → value pairs:

```json
{
  "partition_id": 0,
  "pairs": [["hale kohea", "$195 – $265"], ["hale mahana", "$215 – $265"]],
  "size": 7,
  "num_conflicts_removed": 0
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `partition_id` | int | Partition index. |
| `pairs` | list of `[name, value]` | Matched attribute/value pairs in the partition. |
| `size` | int | Number of pairs (== `len(pairs)`). |
| `num_conflicts_removed` | int | Pairs dropped during conflict resolution. |

## Inspecting the full file (stream, don't load 4 GB)

```bash
cd output/results_1500k_jaccard          # or the dama path above

wc -l resolved_mappings.jsonl                     # record count
sed -n '500p' resolved_mappings.jsonl | python3 -m json.tool   # pretty-print one record
jq .size resolved_mappings.jsonl | sort -n | uniq -c           # partition-size distribution
jq 'select(.size > 50)' resolved_mappings.jsonl                # only large partitions
```
