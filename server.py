"""Demo server for AutoMap pipeline.
Run with: python server.py
"""

from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

DATA_PATH = "dev_chunks/chunk_1_mini_1000.json"
OUTPUT_PATH = "output_1000"

# ---------------------------------------------------------------------------
# Helper to load JSONL
# ---------------------------------------------------------------------------
def load_jsonl(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

# ---------------------------------------------------------------------------
# Route 1: Raw tables (Intro)
# ---------------------------------------------------------------------------
@app.route("/api/raw-tables")
def raw_tables():
    """Return raw tables with highlighted potential mapping columns."""
    tables = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 15:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "relation" not in rec or len(rec["relation"]) < 2:
                continue

            cols = rec["relation"][:4]
            # Find pairs that look like mappings:
            # col where each value is unique (potential left side)
            mapping_hint = None
            for ci in range(len(cols)-1):
                col_a = [v for v in cols[ci] if v]
                col_b = [v for v in cols[ci+1] if v]
                if len(col_a) >= 2 and len(set(col_a)) == len(col_a):
                    mapping_hint = [ci, ci+1]
                    break

            tables.append({
                "title": rec.get("pageTitle", f"Web Table {i+1}") or f"Web Table {i+1}",
                "columns": [c[:5] for c in cols],
                "num_cols": len(rec["relation"]),
                "num_rows": len(rec["relation"][0]) if rec["relation"] else 0,
                "mapping_hint": mapping_hint
            })

            if len(tables) >= 4:
                break

    return jsonify({"tables": tables})

# ---------------------------------------------------------------------------
# Route 2: Step 1 — Candidate Extraction
# ---------------------------------------------------------------------------
@app.route("/api/step1")
def step1():
    """Return WP1 + WP2 results."""
    filtered = load_jsonl(os.path.join(OUTPUT_PATH, "filtered_corpus.jsonl"))

    # Count columns after filtering
    total_cols_after = sum(len(t["relation"]) for t in filtered)

    # Count rejected columns
    total_rejected = sum(len(t.get("rejected_column_indices", [])) for t in filtered)

    # Total before = after + rejected
    total_cols_before = total_cols_after + total_rejected

    # Find a removed column example
    removed_example = None
    kept_example = None

    for t in filtered:
        if t.get("rejected_column_indices") and removed_example is None:
            scores = t.get("coherence_scores", [])
            cols = t["relation"]
            if cols and scores:
                removed_example = {
                    "values": cols[0][:5],
                    "score": round(scores[0], 3)
                }
        if t["relation"] and kept_example is None:
            scores = t.get("coherence_scores", [])
            kept_example = {
                "values": t["relation"][0][:5],
                "score": round(scores[0], 3) if scores else 0.9
            }
        if removed_example and kept_example:
            break

  # Fallback removed example if none found
    # This corpus is already clean (pre-filtered WDC data)
    # so we show a synthetic example of what gets removed
    if not removed_example:
        removed_example = {
            "values": ["2024-01-01", "hello world", "83.5%", "the matrix", "blue"],
            "score": 0.21,
            "is_example": True
        }

    candidates = load_jsonl(os.path.join(OUTPUT_PATH, "candidates.jsonl"))
    sample_candidates = [
        {
            "pairs": c["pairs"][:3],
            "theta": round(c["theta"], 3),
        }
        for c in candidates[:5]
    ]

    return jsonify({
        "wp1": {
            "tables_processed": len(filtered),
            "columns_before": total_cols_before,
            "columns_after": total_cols_after,
            "columns_removed": total_rejected,
            "removed_example": removed_example,
            "kept_example": kept_example,
        },
        "wp2": {
            "candidates_found": len(candidates),
            "sample_candidates": sample_candidates,
        }
    })# ---------------------------------------------------------------------------
# Route 3: Step 2 — Table Synthesis
# ---------------------------------------------------------------------------
@app.route("/api/step2")
def step2():
    """Return WP3 results: synthesized mappings."""
    mappings = load_jsonl(os.path.join(OUTPUT_PATH, "synthesized_mappings.jsonl"))

    total = len(mappings)
    singletons = sum(1 for m in mappings if m["num_source_tables"] == 1)
    multi = total - singletons
    avg_pairs = sum(m["size"] for m in mappings) / total if total else 0

    # Top 6 largest mappings
    sorted_mappings = sorted(mappings, key=lambda x: x["size"], reverse=True)
    top_mappings = [
        {
            "partition_id": m["partition_id"],
            "size": m["size"],
            "num_source_tables": m["num_source_tables"],
            "sample_pairs": m["pairs"][:4],
        }
        for m in sorted_mappings[:6]
    ]

    return jsonify({
        "total_mappings": total,
        "singleton_partitions": singletons,
        "multi_table_partitions": multi,
        "avg_pairs": round(avg_pairs, 1),
        "top_mappings": top_mappings,
    })

# ---------------------------------------------------------------------------
# Route 4: Step 3 — Conflict Resolution
# ---------------------------------------------------------------------------
@app.route("/api/step3")
def step3():
    """Return WP4 results: resolved mappings."""
    resolved = load_jsonl(os.path.join(OUTPUT_PATH, "resolved_mappings.jsonl"))
    synthesized = load_jsonl(os.path.join(OUTPUT_PATH, "synthesized_mappings.jsonl"))

    # Build lookup for synthesized by partition_id
    synth_lookup = {m["partition_id"]: m for m in synthesized}

    total = len(resolved)
    had_conflicts = sum(1 for m in resolved if m["num_conflicts_removed"] > 0)
    total_removed = sum(m["num_conflicts_removed"] for m in resolved)

    # Build before/after examples
    conflict_examples = []
    for m in resolved:
        if m["num_conflicts_removed"] > 0:
            orig = synth_lookup.get(m["partition_id"])
            if orig:
                # Find pairs that were removed (in orig but not in resolved)
                resolved_set = set(tuple(p) for p in m["pairs"])
                removed_pairs = [p for p in orig["pairs"] if tuple(p) not in resolved_set]
                conflict_examples.append({
                    "partition_id": m["partition_id"],
                    "pairs_kept": m["size"],
                    "pairs_removed": m["num_conflicts_removed"],
                    "before_pairs": orig["pairs"][:5],
                    "after_pairs": m["pairs"][:5],
                    "removed_pairs": removed_pairs[:3],
                })
        if len(conflict_examples) >= 4:
            break

    # Top 5 final mappings
    sorted_resolved = sorted(resolved, key=lambda x: x["size"], reverse=True)
    top_mappings = [
        {
            "partition_id": m["partition_id"],
            "size": m["size"],
            "num_conflicts_removed": m["num_conflicts_removed"],
            "sample_pairs": m["pairs"][:3],
        }
        for m in sorted_resolved[:5]
    ]

    return jsonify({
        "total_mappings": total,
        "mappings_with_conflicts": had_conflicts,
        "total_pairs_removed": total_removed,
        "conflict_examples": conflict_examples,
        "top_mappings": top_mappings,
    })

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("AutoMap demo server running at http://localhost:5000")
    app.run(debug=True, port=5000)