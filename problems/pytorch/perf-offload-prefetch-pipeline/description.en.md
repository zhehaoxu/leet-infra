# Offload Prefetch Pipeline: Compute-Transfer Overlap Timeline under a Single Engine and Multiple Buffers

## Problem

When GPU memory runs out, schemes like ZeRO-Offload / ZeRO-Infinity keep parameters or optimizer states in DRAM (or even NVMe) and stream them into HBM layer by layer. If the copy can **overlap** with compute, wall-clock time is "total compute + the transfer time that is not hidden"; the unhidden part is the **stall**.

Whether overlap happens depends on two things: the **copy engine is serial** (it moves one layer at a time), and there are `buffers` staging slots in HBM. With `buffers=1`, the next layer cannot start copying until the current layer finishes computing and frees the slot — no overlap at all; with `buffers >= 2` the engine can copy layer `j+1` while layer `j` computes.

This problem nails down the pipeline on an integer timeline: given per-layer compute time, per-layer transfer bytes, engine bandwidth, and slot count, output the makespan, total stall, and each layer's compute start, prefetch start, and prefetch end.

## Rules

`compute[j]` is the compute time of layer `j`, `transfer_bytes[j]` the bytes to prefetch from DRAM, `bw` the bandwidth (bytes per unit time), and `buffers` the HBM staging slots. One engine, strictly in-order prefetch, compute and transfer may run concurrently:

1. `transfer_time[j] = ceil(transfer_bytes[j] / bw)` (0 bytes costs 0 and does not occupy the engine);
2. layer `j` occupies slot `j % buffers` from **prefetch start** until **compute end**;
3. prefetch start `s_j = max(engine_free, finish[j-buffers])` (the second term is 0 for `j < buffers`): `engine_free` is the end of the previous prefetch (serial engine), `finish[j-buffers]` is when the layer that last used the same slot finished computing;
4. prefetch end `e_j = s_j + transfer_time[j]`;
5. compute start `c_j = max(finish[j-1], e_j)` (`finish[-1] = 0` for the first layer) — it cannot start before the previous layer's compute ends, nor before its own prefetch completes;
6. `finish[j] = c_j + compute[j]`; **stall** is `Σ(c_j - finish[j-1])`; and `makespan = finish[L-1] = Σ compute + stall`.

Return `(makespan, stall_total, compute_start, transfer_start, transfer_end)`, the last three in layer order.

## Example

With `bw=50`, four layers, `compute=[5,3,4,2]`, `transfer_bytes=[100,200,50,150]` → transfer times `[2,4,1,3]`, `buffers=2`:

```
j0: slot 0 free -> s=0  e=2  c=2   finish=7       slot 0 held [0,7)
j1: slot 1 free -> s=2  e=6  c=7   finish=10      slot 1 held [2,10)
j2: slot 0 freed at finish[0]=7 -> s=7  e=8  c=10 finish=14
j3: slot 1 freed at finish[1]=10 -> s=10 e=13 c=14 finish=16
```

```
makespan       = 16        # = Σcompute 14 + stall 2
stall_total    = 2         # only the first layer waits for the first prefetch
compute_start  = [2, 7, 10, 14]
transfer_start = [0, 2, 7, 10]
transfer_end   = [2, 6, 8, 13]
```

For contrast, the same data with `buffers=1`: every layer must wait for the previous one to finish computing before copying, stall becomes `2+4+1+3=10`, and the makespan becomes `24`.

## Constraints

- `compute[j] >= 1`, `transfer_bytes[j] >= 0`, `bw >= 1`, `buffers >= 1`; empty input is allowed (returns `0, 0` and three empty lists).
- Transfer time rounds **up**: 40 bytes at 30 bytes/unit is 2 units, not 1.33.
- Slots are reused only among layers whose index is congruent modulo `buffers`; layers with `j < buffers` can prefetch from time 0 (slots start free).
- A zero-byte layer costs no engine time but still obeys slot rules (`buffers=1` means it cannot even be "issued" until the previous layer finishes).
- Fixed return types: the first two are `int`, the last three are `list[int]`.
