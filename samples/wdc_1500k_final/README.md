# Covered-1.5M capped run — FINAL project results (sample)

This is the project's final run: 1.5M WDC web tables sampled with the
**covered-sample method** (every benchmark cover table included) and scored with
`AUTOMAP_SCORE_PAIR_CAP=150`. Completed 2026-07-06 in 10.5 h.

**Headline: 86.06% covered recall** (25,933 / 30,134 reachable gold pairs) —
flat vs. the covered-100k baseline (86.2%), i.e. recall is insensitive to the
scoring pair cap and to corpus scale.

See **`RESULTS.md`** (copied from `dama:/home/automap/Automap/results/`) for the
full manifest: exact invocation, all configs, per-stage timings, resource
trace, and the 100k-vs-1.5M comparison.

## Files

| File | What it is |
|------|------------|
| `RESULTS.md` | full run manifest (configs, timings, eval, provenance) |
| `resolved_mappings.sample.jsonl` | 1,000 records **randomly sampled** (`shuf -n 1000`, reservoir) from the final 3.5 GB / 4,144,953-record output |
| `synthesized_mappings.sample.jsonl` | 1,000 random records of the pre-conflict-resolution output |
| `benchmark_eval.txt` | full `eval_benchmark.py` output (global / achievable / covered recall, scoped precision) |
| `threshold_sweep.txt` | Stage 4 PMI threshold sweep |
| `coherence_distribution.png` | Stage 3 coherence score distribution |

Unlike the older samples (first-1,000 records), these are uniform random
samples, so partition sizes and junk-blob frequency are representative.

## Full results location (not in git)

- **dama:** `/home/automap/Automap/results/` (canonical: final
  `resolved_mappings.jsonl` 3.5 GB, full pipeline log, memory traces, driver)
- Intermediates (candidates 17 GB, edge scores, filtered corpus) in
  `dama:/home/automap/Automap/output/wdc_covered_1500k_capped/`
