"""Run only WP4 (conflict resolution) on an already-saved
synthesized_mappings.jsonl.

Use after a WP3 run where Stage 7 was killed (e.g. due to the O(P^3) bug
that has since been fixed) and we want to recover the WP4 output without
re-running WP1-WP3.

Usage:
    python run_wp4.py \\
        --synthesized output/<folder>/synthesized_mappings.jsonl \\
        --resolved   output/<folder>/resolved_mappings.jsonl
"""

from __future__ import annotations

import argparse
import time

from conflict_resolution import run_conflict_resolution
from heartbeat import heartbeat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthesized", required=True,
                   help="path to synthesized_mappings.jsonl from WP3")
    p.add_argument("--resolved", required=True,
                   help="path to write resolved_mappings.jsonl")
    args = p.parse_args()

    t0 = time.time()
    print(f"[WP4] Conflict resolution from {args.synthesized}")
    with heartbeat("wp4-only"):
        run_conflict_resolution(args.synthesized, args.resolved)
    print(f"  Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
