# what we missed in the paper (and what i'm gonna fix)

so after the 10k vertica run kept blowing up (4-day ETA on the merge loop), i went back and re-read the wang+he automap paper end-to-end. turns out we've been missing 3 things that the paper explicitly mentions as scalability tricks, including one really big one. that's why we couldn't scale.

context: the paper runs this on **100 million tables** (way more than the entire vertica corpus we have access to) and finishes in ~20 hours on map-reduce. so it's not a hardware thing. without these tricks none of their experiments would have completed.

> heads-up to anyone reading an earlier draft of this: i initially also called out our `positive_score` formula as buggy. that was wrong — i re-checked the code and we already implement equation 3 (max-of-containment) correctly. the `last_w+=165.833` numbers in the merge log come from the **post-merge aggregation step** (the code sums the constituent edge weights when partitions merge, which can grow >1), not from the initial scoring. so the formula itself is fine.

---

## 1. we don't filter low-quality positive edges (θ_edge)

paper section 5.4:

> "θ_edge is the threshold to filter out edges with insignificant positive weight. our experiment suggests that θ_edge = 0.85 has the best performance."

we keep any positive intersection. they drop everything below 0.85. since our initial `w⁺` is already correctly normalized to [0, 1], θ_edge=0.85 means: only keep edges where the shared pairs are >= 85% of the smaller table. that single filter kills most of the noise positives. for context: yesterday's 10k run had **70 million** positive edges. with θ_edge=0.85 probably 90%+ of those get dropped.

## 2. we don't threshold the overlap for negative edges (θ_overlap default is too low)

paper section 4.1:

> "we evaluate w⁻(B, B') only if B and B' share more than θ_overlap left-hand-side values"

and section 5.4:

> "as θ_overlap increases, |E| drops quickly. the quality of resulting clusters are insensitive to θ_overlap."

we have the flag (`--theta_overlap`) but the default is 1, the lowest possible. they bump it up to drop spurious overlap pairs from common values. our 10k run enumerated **333 million** overlap edges, mostly from common left values like dates and abbrevs. raising θ_overlap to 3 or 5 would cut that by orders of magnitude with basically no quality loss per their sensitivity analysis.

## 3. THE BIG ONE: connected component decomposition (§4.2 + appendix E)

this is the actual scaling technique they describe. literal quote:

> "we use a divide-and-conquer approach to first produce components that are connected non-trivially by positive edges on the full graph, and then look at each subgraph individually"

the logic: two candidates with no positive-edge path between them can **never** end up in the same partition (the algorithm only merges things with w⁺ > 0). so their scores never need to be considered together. instead of one giant merge loop scanning all 70M positive edges per iteration, you run **thousands of tiny merge loops, one per connected component**, each scanning only that component's edges.

this is why our 10k run was going to take 4 days — every merge round did an O(70M) scan, which is ~5 seconds in python. with component splitting, each round scans maybe hundreds to thousands of entries, so basically free. and components are embarrassingly parallel — they can run on different cores at once.

## (bonus) union-find for partition membership

they use a disjoint-set data structure for set union/lookup in the merge loop. we use dict-of-lists with manual concatenation. less critical than the other 3 but standard once we're refactoring.

---

## what i'll fix next

in dependency order, with the components fix being the big one:

1. **add `--theta_edge` flag**, default 0.85, drop edges with w⁺ below it
2. **change default `--theta_overlap`** from 1 to something like 3 (we already have the flag)
3. **split the merge loop by connected components** of the positive-edge graph — this is the architectural one, ~half-day of spec+code, runs each component independently (parallelizable)

verifying on the WDC corpus first (we have known sequential output to compare against) before re-running on vertica. once these land i think the 10k vertica run will go from "days" to "tens of minutes". 100k may even be feasible after this.

i'll follow the same workflow as the parallelization (spec → plan → subagent-driven dev with tests), so the changes should be solid before they touch main.
