# Approach

## What the interviewer is after

"Why does training a 7B model take hundreds of GB of memory when inference only needs a dozen or so GB?"
— the answer lies in **training state**. Inference only needs the weights; during training every parameter
sits in memory together with its gradient, an fp32 master weight, and Adam's two moments. ZeRO is about
slicing these "copies" across GPUs. This problem asks you to recall where **16 bytes/parameter** comes from
and to compute the per-GPU footprint of each sharding stage.

## Where the 16 bytes come from

In mixed-precision training (fp16/bf16 compute + fp32 optimization), one parameter corresponds to:

- **fp16 parameter**: used in forward/backward, 2 bytes;
- **fp16 gradient**: produced by backward, 2 bytes;
- **fp32 master weight**: the high-precision copy maintained by the optimizer, 4 bytes;
- **Adam m / v**: first and second moments, both fp32, 4 bytes each.

Total $2+2+4+4+4 = 16$ bytes/parameter. So the pure training state (excluding activations) is about $16N$:
a 7B model ≈ $112$GB, which is exactly the motivation for "does not fit on one GPU, must shard".

## What each of the three ZeRO stages shards

ZeRO's core insight: in data parallelism every GPU redundantly stores the full set of states, and these
redundancies can be split apart on demand.

- **Stage 1** shards the **optimizer state** (master+m+v = 12 bytes) → each GPU keeps the replicated
  param+grad (4 bytes) plus $\frac{12}{W}$.
- **Stage 2** additionally shards the **gradients** (+2) → the replicated part is only param (2 bytes), and
  the sharded part is $\frac{14}{W}$.
- **Stage 3** shards even the **parameters** → the full set is $\frac{16}{W}$.

Substituting $W=1$ should degenerate to $16N$ (nothing to shard), which is a self-check; the test
`check_world_one_is_full` verifies exactly this.

## Intuition and traps

- Higher stages save more memory but cost more communication/recomputation (Stage 3 must temporarily
  all-gather parameters during the forward pass).
- Do not count the "fp16 parameter" as 4 fp32 bytes — that would make the total 18.
- Activation memory is outside the scope here; real OOMs are often triggered first by activations
  (especially with long sequences), see `transformer-mem`.

## Judging methodology

`test_cases.py` cross-checks the simplified closed form against an independent reference `_ref` that
**assembles the terms one by one** (first computing the $N$-scaled param/grad/master/m/v terms, then summing
them per stage); the answer passes only when both paths agree. Floats use a $10^{-6}$ relative tolerance.
