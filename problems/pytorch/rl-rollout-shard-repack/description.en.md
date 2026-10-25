# RL Systems: Rollout Re-sharding Across Ranks and Padding-Free Packing

## Problem

In frameworks like verl / OpenRLHF, the data of one PPO step is re-partitioned twice: the generation side (vLLM) lays out its $N$ rollouts according to its own sharding, while the $W$ data-parallel ranks on the training side demand a **token-balanced** split — otherwise the rank holding a few long-tail sequences becomes the straggler of the whole step (this is exactly what verl's `rebalance_batch` and DAPO's global partition do). After balancing, each rank packs its variable-length sequences **padding-free**: flash-attention's varlen interface chains several sequences into one micro-batch, and compared with the classical "pad every sequence to the rank's longest" scheme, the pad tokens saved are this step's net win.

This problem touches no tensor contents at all — it is pure **index and integer bookkeeping**.

## Rules

Three inputs: `seq_lens` (int64 $(N,)$, token count of each rollout sequence, $\ge 1$, listed in generation-shard order), `num_ranks` ($W$), `micro_budget` ($M$, the token budget of one micro-batch).

1. **Initial ownership**: sequence $i$ starts on rank $\text{init}(i)=\lfloor i\cdot W/N\rfloor$ (contiguous blocks, not round-robin).
2. **Re-sharding**: the processing order is sequences sorted ascending by $(\text{length},\ \text{original index})$; each sequence in turn goes to the rank with the **smallest current real token total** (ties → smallest rank index). This gives `dest_rank`.
3. **Micro-batch packing**: inside each rank, greedily pack following the order of step 2 — if the open pack is **non-empty** and adding this sequence would exceed $M$, close it and open a new one; a single sequence longer than $M$ **occupies its own pack** (that pack may exceed the budget; sequences cannot be split). Micro-batch ids are numbered globally and contiguously from $0$, ranks in ascending order, within a rank in order of appearance. This gives `micro_batch`.
4. **Cost accounting**: `tokens_per_rank[r]` $=$ the real token sum of rank $r$ (the padding-free cost); `pad_saved[r]` $=$ the rank's sequence count $\times$ its longest sequence $-$ its real token sum, i.e. the extra tokens classical sequence-level padding would compute. `moved[i]` $=1$ when `dest_rank[i] != init(i)`, else $0$ — it counts how many sequences have to be transferred across ranks.

Return a 5-tuple (all `int64`): `dest_rank` $(N,)$, `micro_batch` $(N,)$, `tokens_per_rank` $(W,)$, `pad_saved` $(W,)$, `moved` $(N,)$.

## Example

`seq_lens=[30,10,50,20,40,60,15,25]` ($N=8$, total tokens $=250$), `num_ranks=3`, `micro_budget=70`:

```
balanced order (length asc, ties by index): 1(10) 6(15) 3(20) 7(25) 0(30) 4(40) 2(50) 5(60)
dest_rank       = [1, 0, 0, 2, 2, 1, 1, 0]
tokens_per_rank = [85, 105, 60]                 # 85+105+60 = 250

rank0 gets {1,7,2}: mb0=[10+25=35], mb1=[50]     # 35+50=85 > 70 → open a new pack
rank1 gets {6,0,5}: mb2=[15+30=45], mb3=[60]
rank2 gets {3,4}  : mb4=[20+40=60]
micro_batch       = [2, 0, 1, 4, 4, 3, 2, 0]     # 5 micro-batches, ids 0..4

sequence-level padding cost: rank0 3 seqs padded to 50 → 150, rank1 → 180, rank2 → 80 (total 410)
pad_saved         = [65, 75, 20]                 # 410 − 250 = 160 pad tokens saved

initial ownership init = [0, 0, 0, 1, 1, 1, 2, 2]
moved             = [1, 0, 0, 1, 1, 0, 1, 1]     # 6 sequences must move across ranks
```

## Constraints

- CPU; $N\le 4096$, $1\le W\le 64$, $M\ge 1$, `seq_lens[i] ≥ 1`; all five outputs are `int64`, compared with `torch.equal`.
- Invariants: $\sum_r$ `tokens_per_rank[r]` $=\sum_i$ `seq_lens[i]`; `pad_saved[r]` $\ge 0$ (exactly $0$ when all lengths in that rank are equal); the ids used by `micro_batch` are precisely the contiguous set $\{0,\dots,B-1\}$; every micro-batch's token sum is $\le M$ unless it holds a single sequence; an empty rank has `tokens_per_rank`/`pad_saved` equal to $0$ and produces no micro-batch.
- Edges: with $W=1$ every sequence lands on rank 0 and `moved` is all $0$; with $W>N$ at most $N$ ranks receive sequences; $N=0$ gives empty `dest_rank`/`micro_batch`/`moved` and length-$W$ zero vectors for `tokens_per_rank`/`pad_saved`; when $M$ is at least a rank's total tokens that rank has exactly one micro-batch.
- Division of labour: `sequence-pack-pad` packs fixed-length blocks **inside one stream** and emits token-level `position_ids`/`segment_ids`; this problem emits the **cross-rank ownership, micro-batch ids, and the cost delta between the two padding schemes** without producing any token-level tensor. Loss/advantage formulas live in the `to-*` collection, communication accounting in `distributed-*`.
