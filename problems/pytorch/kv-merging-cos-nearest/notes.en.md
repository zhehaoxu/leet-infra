# Solution Notes · Merging

## The Core 5 Steps

1. topk picks the kept slots; the complement via nonzero gives the evicted ones
2. Normalize the keys on both sides → cosine similarity matrix
3. For each evict, find the most similar kept slot (max-sim)
4. Weight = similarity × norm ratio
5. scatter_add into the target slot → normalize → write back

## Why Use the Norm Ratio

Weighting by sim alone lets "a large value drag a small value off-center" — for example, when $\|v_\text{evict}\| = 100, \|v_\text{target}\| = 1$, the merged result would be swamped by the evict. The norm ratio $e / (e + t)$ acts like a **soft sigmoid**: when the evict is relatively weak it gets a high weight (aggregating the weaker neighbor's information), and when the evict is stronger its weight is diluted (avoiding domination).

## similarity_threshold and merge_fraction

Real kvpress allows two extra gates:
- `similarity_threshold > 0`: only merge when the similarity is high enough
- `merge_fraction < 1`: only merge the evicts in the top-$\alpha$ similarities

This problem fixes `threshold=0.0, fraction=1.0` for simplicity — but still keeps `max_sim.clamp(min=0)`: negative sims are discarded outright (equivalent to threshold=0).

## Only Rewrite Kept Slots, Leave Evicted Slots Alone

`result = values.clone(); result.scatter_(1, keep_idx, updated)` — the values at evicted positions numerically "stay in place" but get filtered out by downstream topk. The problem returns the full shape; when judging, kept positions are compared against the merged results and evicted positions against their original values.

## Relation to CAM / D2O

CAM also "merges instead of pruning", using attention-based weighting; D2O uses output deviation to decide the merge target. Merging uses **cosine similarity** — the simplest approach, and the prototype method of this family.
