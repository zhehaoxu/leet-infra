# ZeRO-Infinity Three-Tier Offload Accounting Explained

## The Two Upgrades from Offload to Infinity

1. **One more storage medium**: ZeRO-Offload has two tiers, HBM↔DRAM; Infinity adds **NVMe SSD**, forming three tiers: HBM (fastest, smallest) → DRAM → NVMe (slowest, largest, "infinite"). Data such as optimizer states—"large and infrequently accessed"—is best placed deep on SSD, leaving GPU memory for only the shards actually needed right now.
2. **Offloaded data is also sharded along DP**: this is the key difference between Infinity and Offload. In ZeRO-Offload, **every data-parallel rank offloads the complete optimizer state to its own CPU** (the offloaded volume is independent of $d$); Infinity **slices the offloaded data and distributes it across ranks**, with each rank storing only $1/d$. Hence in this problem $P,G,O$ are all divided by $d$—including the part offloaded to DRAM/SSD.

## How the Three-Tier Ledger Is Laid Out

The model is deliberately simplified to "prefetch the entire offloaded shard in one step":

- HBM: the gradient shard $G$ (used immediately each step, never offloaded) + the non-offloaded parameter/optimizer shards.
- Offloaded shards $= \mathbb{1}[\text{offload\_param}]P + \mathbb{1}[\text{offload\_opt}]O$, all placed in the target tier (dram or ssd), with the other tier at 0.
- `step_transfer_bytes` = the offloaded shards: the volume moved back from the lower tier into HBM each step (a measure of the pressure on PCIe/NVMe bandwidth).

## Edge Cases and Intuition

- Both switches False → degenerates to **ZeRO-3 fully in HBM**: `hbm_bytes=(P+G+O)=16N/d`, transfer 0. Useful as a reference for "how much memory is needed without offloading."
- Offload only the optimizer (the most common Infinity configuration) → HBM saves the largest term $O=12N/d$, while $P,G$ still reside there.
- Whether target is dram or ssd **only changes which tier the footprint lands in**, not the HBM usage or the total transfer volume—in a real system the difference is bandwidth/latency (NVMe is slower and needs `afr` prefetching), which this problem ignores.

## Implementation Pitfalls

- Do not forget to **divide everything by $d$** (including the offloaded part); this is the point that makes Infinity≠Offload. Writing the full amounts makes the result noticeably too large.
- Use float division for $P,G,O$ to avoid integer truncation; return everything as `float`.
- dram/ssd are mutually exclusive: the tier that is not the target must be 0, not "hold a copy as well."

## Related

- Compare: `distributed-zero-offload` (two tiers, offloaded data not sharded by $d$), `distributed-zero-communication` (partitioned communication volume), `training-state-memory` (the 16-bytes-per-parameter baseline).
- Judging: a pure closed form plus an independent recomputation in `test_cases`, with relative error $10^{-6}$.
