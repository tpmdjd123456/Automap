# what we missed in the paper

re-read wang+he end to end. found 3 scalability tricks we'd missed:

1. **θ_edge=0.85** — drop weak positive edges (paper §5.4)
2. **θ_overlap=3** — drop spurious overlap from common values (paper §4.1)
3. **connected-component decomposition of the merge loop** (paper §4.2 + App. E) — many tiny merge loops instead of one giant scan. this is the big one.

**verified on dama:** WDC 102 min → 2.3 min (44×). Vertica 10k from a 4-day projection → 11 min.

full corpus (~116M filtered tables) is still infeasible on a single machine: the overlap-set Counter would need TBs of RAM. paper handles that with Map-Reduce. our ceiling on dama is ~1M tables.

pushed. running 1M overnight.
