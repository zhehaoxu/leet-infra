# OOM Boundary Search: Inverting Peak Memory for Micro-Batch and Sequence Length

## Problem

When tuning batch sizes engineers face two inverse questions: **the sequence length is fixed - how large can the
micro batch still be? the micro batch is fixed - how long can the sequence get?** Peak memory holds a
**static footprint** (parameters, gradients, optimizer state) that ignores the batch, plus activations made of a
"per micro-batch per token" **linear term** and an attention-score term of **$O(s^2)$**. The first is linear in $m$,
the second quadratic in $s$, so the two boundaries behave very differently: one is an integer floor division, the
other the positive root of a quadratic - and one step in the rounding direction is the difference between "fits
exactly" and OOM. Once the boundaries are known there is still the question of how many data-parallel ranks that
global batch needs, and whether it divides at all.

`transformer-mem` / `training-state-memory` give the **byte counts themselves** (no inversion against batch),
`distributed-zero-shard-ownership` handles shard ownership, `kv-cache-memory` handles the inference cache; this
problem is only about **searching outward for the largest feasible configuration under a byte budget**, plus the
integrality test that a data-parallel degree must pass.

## Rules

- **Budget** $\text{usable}=\texttt{hbm\_bytes}-\texttt{reserve\_bytes}$: the reserve is allocator-fragmentation /
  driver headroom and is **not part of the peak**, whereas `cuda_overhead_bytes` **is**; `usable` may be negative.
- **Static footprint** $\text{static}=\text{param}+\text{grad}+\text{optim}$ (ZeRO-0: every rank keeps a full copy).
  The default cases split it per `training-state-memory`'s 16 bytes/parameter (fp16 param 2 + fp16 grad 2 + fp32
  master/m/v 12).
- **Peak** $\text{peak}(m,s)=\text{cuda\_overhead}+\text{static}+m\cdot s\cdot\texttt{act\_per\_token}+m\cdot s^2\cdot\texttt{act\_per\_token\_sq}$ - the quadratic term is multiplied by $m$ too, since every sequence carries its own $s^2$ score block.
- **Feasibility is inclusive**: $\text{fits}\iff \text{peak}\le\text{usable}$ (using memory exactly to the last byte
  still counts), and $\text{headroom}=\text{usable}-\text{peak}$, **which may be negative and is not clamped**.
- **The two boundaries**: the largest integer $m\ge1$ at fixed $s$, and the largest integer $s\ge1$ at fixed $m$;
  report $0$ when even $1$ does not fit. The $m$ boundary is linear, so
  $\lfloor \text{budget}/(s\cdot ap+s^2\cdot apq)\rfloor$; the $s$ boundary solves
  $m\,apq\,s^2+m\,ap\,s\le\text{budget}$ (with $\text{budget}=\text{usable}-\text{cuda\_overhead}-\text{static}$) -
  take the floor of the positive root and correct it against the predicate. With
  $\texttt{act\_per\_token\_sq}=0$ it degenerates to a linear floor.
- **Data parallelism** $\text{world}=\lceil G/(A\cdot m)\rceil$ uses the **given** `micro_batch` (not
  $\text{max\_micro\_batch}$), and $\text{global\_batch\_fits}=\text{fits}\wedge(\text{world}\cdot A\cdot m=G)$:
  the rank count must be an integer and the last micro-batch may not be half empty, so a $G$ that is not divisible
  by $A\cdot m$ fails.
- Input meanings, every formula and the return typing live in the `starter.py` docstring, the single source of truth.

## Examples

```
Input: hbm=80e9, reserve=2e9, cuda_overhead=1.5e9, param=2.6e9, grad=2.6e9, optim=15.6e9,
       act_per_token=1.5e6, act_per_token_sq=128, micro_batch=4, seq=2048, global_batch=512, accum=8
  usable = 78e9   static = 20.8e9   budget = 78e9-1.5e9-20.8e9 = 55.7e9
  peak(4,2048) = 1.5e9+20.8e9+4*2048*1.5e6+4*2048^2*128 = 36,735,483,648 (12.288GB linear + 2.147GB quadratic)
  headroom = 41,264,516,352   feasible = True
  max_micro_batch = floor(55.7e9 / (2048*1.5e6+2048^2*128)) = floor(55.7e9/3,608,870,912) = 15
  max_seq: floor of the positive root of 512 s^2 + 6e6 s <= 55.7e9 = 6103
  world = ceil(512/(8*4)) = 16, and 16*8*4 = 512 -> global_batch_fits = True
Output: (20800000000, 36735483648, 41264516352, True, 15, 6103, 16, True)

Input: same but micro_batch=8, seq=8192
  peak(8,8192) = 1.5e9+20.8e9+8*8192*1.5e6+8*8192^2*128 = 189,323,476,736 > 78e9 -> OOM
  headroom = -111,323,476,736   feasible = False
  max_micro_batch = floor(55.7e9/20,877,934,592) = 2   max_seq (at m=8) = 3560
  world = ceil(512/(8*8)) = 8 and 8*8*8 = 512 divides, but the memory does not fit -> False
Output: (20800000000, 189323476736, -111323476736, False, 2, 3560, 8, False)

Input: hbm=20000, reserve=cuda_overhead=param=grad=optim=0, act_per_token=1000,
       act_per_token_sq=0, micro_batch=2, seq=10, global_batch=40, accum=4
  peak = 2*10*1000 = 20000 == usable -> inclusive test passes, headroom = 0
  max_micro_batch = 20000//(10*1000) = 2   max_seq = 20000//(2*1000) = 10
  world = ceil(40/(4*2)) = 5, and 5*4*2 = 40 -> True
Output: (0, 20000, 0, True, 2, 10, 5, True)
(Change hbm to 19999: feasible=False, headroom=-1, and both boundaries drop one step to 1 and 9)
```

## Constraints

- `hbm_bytes` $>0$; `reserve_bytes`, `cuda_overhead_bytes`, `param_bytes`, `grad_bytes`, `optim_bytes` $\ge 0$;
  `act_per_token` $\ge 1$; `act_per_token_sq` $\ge 0$; `micro_batch`, `seq`, `global_batch`, `accum_steps` $\ge 1$
- Return the 8-tuple `(static_bytes, peak_bytes, headroom_bytes, feasible, max_micro_batch, max_seq,
  world_size_needed, global_batch_fits)`: items 1, 2, 3, 5, 6, 7 are exact ints, items 4 and 8 are bools
- Everything is integer byte arithmetic - no floating-point comparisons in the judge; `act_per_token >= 1` keeps
  both boundaries bounded, so besides "0 when even 1 does not fit" there are no sentinel values
