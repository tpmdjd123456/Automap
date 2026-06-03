"""End-to-end pipeline driver for Section 3 of Wang & He (SIGMOD 2017):
Candidate Table Extraction.

Stages:
  1. Load corpus (JSONL or CSV folder)
  2. Build (or load cached) co-occurrence index
  3. Score every column for coherence (NPMI)
  4. Filter columns at threshold; emit JSONL + histogram + sweep table   [WP1]
  5. Approximate-FD filter on ordered column pairs; emit candidates       [WP2]
  6. Greedy table synthesis; group candidates into mappings               [WP3]

Run:
    python main.py --corpus_path data/sample.json \
                   --output_folder output/ \
                   --threshold 0.3 \
                   --theta 0.95 \
                   --index_path output/cooccurrence_index.pkl
"""

from __future__ import annotations

import argparse
import os
import time

from data_loader import load_corpus, corpus_summary
from cooccurrence_index import (
    build_cooccurrence_index,
    save_index,
    load_index,
    index_summary,
)
from npmi import score_corpus, test_npmi as npmi_sanity
from filter import (
    filter_corpus,
    rebuild_filtered_corpus,
    save_filtered_corpus,
    filtering_report,
    threshold_sweep,
    plot_coherence_distribution,
)
from fd_filter import (
    filter_candidates_by_fd,
    save_candidates,
    candidates_summary,
)
from synthesis import (
    load_candidates as load_wp3_candidates,
    greedy_partition,
    save_synthesized_mappings,
    synthesis_report,
)

from conflict_resolution import (
    run_conflict_resolution,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Section 3 pipeline: PMI coherence filtering (WP1) "
                    "+ approximate-FD column-pair filtering (WP2)"
    )
    p.add_argument("--corpus_path", required=True,
                   help="Path to JSONL file or folder of CSVs")
    p.add_argument("--output_folder", required=True,
                   help="Where to write filtered corpus, reports, and candidates")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="WP1 coherence threshold (default 0.3)")
    p.add_argument("--theta", type=float, default=0.95,
                   help="WP2 approximate-FD threshold (default 0.95)")
    p.add_argument("--min_rows", type=int, default=3,
                   help="WP2 minimum non-empty rows for a pair to be evaluated "
                        "(default 3)")
    p.add_argument("--index_path", default=None,
                   help="Path to save/load co-occurrence index "
                        "(default: <output_folder>/cooccurrence_index.pkl)")
    p.add_argument("--rebuild_index", action="store_true",
                   help="Force rebuilding the index even if cached")
    p.add_argument("--table_types", default="RELATION",
                   help="Comma-separated tableType values to keep (JSONL only). "
                        "Default: RELATION")
    p.add_argument("--tau", type=float, default=-0.2,
                   help="WP3 negative-weight threshold (default -0.2)")
    p.add_argument("--no_approx", action="store_true",
                   help="WP3 disable approximate string matching")
    p.add_argument("--theta_overlap", type=int, default=1,
                   help="WP3 minimum shared pairs to consider a candidate pair "
                        "(default 1)")
    p.add_argument("--parallel_workers", type=int, default=1,
                   help="Run WP3 initial scoring in parallel with N workers "
                        "(default 1 = sequential). Recommended on dama: 14 "
                        "(one per physical core); on laptop: 6.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    index_path = args.index_path or os.path.join(
        args.output_folder, "cooccurrence_index.pkl"
    )
    table_types = tuple(t.strip() for t in args.table_types.split(",") if t.strip())

    total_start = time.time()

    # ---- Stage 1: Load ------------------------------------------------------
    print(f"[Stage 1/6] Loading corpus from {args.corpus_path}...")
    t0 = time.time()
    corpus = load_corpus(args.corpus_path, table_types=table_types)
    corpus_summary(corpus)
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 2: Index -----------------------------------------------------
    print("[Stage 2/6] Building co-occurrence index...")
    t0 = time.time()
    if (not args.rebuild_index) and os.path.exists(index_path):
        print(f"  Loading cached index from {index_path}")
        index = load_index(index_path)
    else:
        index = build_cooccurrence_index(corpus)
        save_index(index, index_path)
        print(f"  Saved index to {index_path}")
    index_summary(index)
    npmi_sanity(index)
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 3: Score -----------------------------------------------------
    print("[Stage 3/6] Computing coherence scores...")
    t0 = time.time()
    scored = score_corpus(corpus, index)
    if scored:
        avg = sum(s for _, _, _, s in scored) / len(scored)
        top = max(scored, key=lambda x: x[3])
        bot = min(scored, key=lambda x: x[3])
        print(f"  Scored {len(scored)} columns")
        print(f"  Average coherence score: {avg:.3f}")
        print(f"  Highest: {top[2][:5]}{'...' if len(top[2]) > 5 else ''} -> {top[3]:.3f}")
        print(f"  Lowest:  {bot[2][:5]}{'...' if len(bot[2]) > 5 else ''} -> {bot[3]:.3f}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 4: Filter (WP1) ----------------------------------------------
    print("[Stage 4/6] Filtering columns (PMI coherence)...")
    t0 = time.time()
    kept, removed = filter_corpus(scored, threshold=args.threshold)
    filtering_report(kept, removed)
    filtered = rebuild_filtered_corpus(corpus, kept)
    out_jsonl = os.path.join(args.output_folder, "filtered_corpus.jsonl")
    save_filtered_corpus(filtered, out_jsonl)
    print(f"  Saved filtered corpus to {out_jsonl}")
    plot_path = os.path.join(args.output_folder, "coherence_distribution.png")
    plot_coherence_distribution(scored, threshold=args.threshold, output_path=plot_path)
    print(f"  Saved histogram to {plot_path}")
    sweep_path = os.path.join(args.output_folder, "threshold_sweep.txt")
    threshold_sweep(scored, output_path=sweep_path)
    print(f"  Saved threshold sweep to {sweep_path}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 5: Approximate-FD column-pair filter (WP2) -------------------
    print(f"[Stage 5/6] FD filtering (theta={args.theta}, "
          f"min_rows={args.min_rows})...")
    t0 = time.time()
    # Reconstruct filtered_corpus.jsonl record dicts in memory (avoids re-reading
    # the file we just wrote).
    filtered_records = [
        {
            **metadata,
            "relation": columns,
            "coherence_scores": scores,
            "rejected_column_indices": rejected,
        }
        for metadata, columns, scores, rejected in filtered
    ]
    candidates = filter_candidates_by_fd(
        filtered_records, theta_threshold=args.theta, min_rows=args.min_rows
    )
    candidates_summary(candidates)
    candidates_path = os.path.join(args.output_folder, "candidates.jsonl")
    save_candidates(candidates, candidates_path)
    print(f"  Saved {len(candidates)} candidates to {candidates_path}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    # ---- Stage 6: Table Synthesis (WP3) ----------------------------------
    print(f"[Stage 6/6] Table synthesis (tau={args.tau})...")
    t0 = time.time()
    wp3_candidates = load_wp3_candidates(candidates_path)
    print(f"  Loaded {len(wp3_candidates)} candidates")
    if args.parallel_workers > 1:
        from parallel_pipeline import parallel_greedy_partition
        partitions = parallel_greedy_partition(
            wp3_candidates,
            tau=args.tau,
            theta_overlap=args.theta_overlap,
            use_approx=not args.no_approx,
            n_workers=args.parallel_workers,
            output_folder=args.output_folder,
        )
    else:
        partitions = greedy_partition(
            wp3_candidates,
            tau=args.tau,
            theta_overlap=args.theta_overlap,
            use_approx=not args.no_approx,
            output_folder=args.output_folder,
        )
    synthesis_report(partitions, wp3_candidates)
    mappings_path = os.path.join(args.output_folder, "synthesized_mappings.jsonl")
    save_synthesized_mappings(partitions, wp3_candidates, mappings_path)
    print(f"  Saved {len(partitions)} synthesized mappings to {mappings_path}")
    print(f"  Time: {time.time() - t0:.2f}s\n")

    print("WP3 Complete! Synthesized mappings ready for WP4 (conflict resolution).")

    # ---- Stage 7: Conflict Resolution (WP4) ----------------------------------
    print(f"[Stage 7/7] Conflict resolution...")
    t0 = time.time()
    resolved_path = os.path.join(args.output_folder, "resolved_mappings.jsonl")
    run_conflict_resolution(mappings_path, resolved_path)
    print(f" Time: {time.time() - t0:.2f}s\n")

    print("All stages complete!")
    print(f"Total time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
