"""Precision / recall / F1 of pipeline mappings against a ground-truth benchmark.

Benchmark format: <relation>\t<value_a>\t<value_b> per line.
Predictions: resolved_mappings.jsonl (each record has a "pairs" list).

Pairs are compared UNORDERED (the pipeline emits both directions), as DISTINCT
sets, with the same clean_value normalization the pipeline applied to corpus
values. We report:
  - Global P/R/F1 over all distinct pairs.
  - Achievable recall: gold pairs whose BOTH values exist in our loaded subset
    (separates "pipeline missed it" from "data not in this 100k subset").
  - Scoped precision: predicted pairs whose BOTH values are in the benchmark's
    value universe (the region the benchmark actually has an opinion about).
"""

import json
import sys

from data_loader import clean_value, load_corpus

BENCH = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark-web.txt"
PRED = sys.argv[2] if len(sys.argv) > 2 else "output/wdc_run_100k/resolved_mappings.jsonl"
CORPUS = sys.argv[3] if len(sys.argv) > 3 else "data/wdc_100k.jsonl"


def und(a, b):
    return frozenset((a, b)) if a != b else None


# ---- ground truth ----
G = set()
V_bench = set()
with open(BENCH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        a, b = clean_value(parts[1]), clean_value(parts[2])
        if not a or not b:
            continue
        p = und(a, b)
        if p:
            G.add(p)
            V_bench.add(a)
            V_bench.add(b)

# ---- predictions ----
P = set()
with open(PRED, encoding="utf-8", errors="ignore") as f:
    for line in f:
        r = json.loads(line)
        for a, b in r["pairs"]:
            a, b = clean_value(a), clean_value(b)
            p = und(a, b)
            if p:
                P.add(p)

# ---- corpus value universe (for achievable recall) ----
V_corpus = set()
for _, cols in load_corpus(CORPUS):
    for col in cols:
        for v in col:
            if v:
                V_corpus.add(v)


def prf(tp, npred, ngold):
    prec = tp / npred if npred else 0.0
    rec = tp / ngold if ngold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


TP = len(P & G)
prec, rec, f1 = prf(TP, len(P), len(G))

G_ach = {p for p in G if all(v in V_corpus for v in p)}
rec_ach = len(P & G_ach) / len(G_ach) if G_ach else 0.0

P_scoped = {p for p in P if all(v in V_bench for v in p)}
tp_sc = len(P_scoped & G)
prec_sc = tp_sc / len(P_scoped) if P_scoped else 0.0

print("Gold distinct pairs (unordered):     %d" % len(G))
print("Predicted distinct pairs (unordered):%d" % len(P))
print("Corpus distinct values (subset):     %d" % len(V_corpus))
print()
print("True positives (P & G):              %d" % TP)
print()
print("=== Global (all predicted pairs) ===")
print("  Precision: %.4f" % prec)
print("  Recall:    %.4f" % rec)
print("  F1:        %.4f" % f1)
print()
print("=== Achievable recall (gold pairs whose both values are in the subset) ===")
print("  Gold pairs present in subset: %d / %d" % (len(G_ach), len(G)))
print("  Achievable recall:            %.4f" % rec_ach)
print()
print("=== Scoped precision (predicted pairs within benchmark value universe) ===")
print("  Predicted pairs in scope: %d" % len(P_scoped))
print("  Scoped TP:                %d" % tp_sc)
print("  Scoped precision:         %.4f" % prec_sc)
print()
print("=== Scoped F1 (scoped precision + achievable recall) ===")
_, _, f1_sc = prf(len(P_scoped & G_ach), len(P_scoped), len(G_ach))
print("  Scoped F1: %.4f" % f1_sc)
