# Per-Layer Communication Volume of Tensor Parallelism (Megatron TP)

## Problem

Data parallelism synchronizes **gradients** each step (size $\propto$ parameter count), whereas tensor parallelism (Tensor Parallel, TP) synchronizes
**activations** at every layer (size $\propto$ batch×seq×hidden)—it splits a single layer's large matrices by column/row across GPUs, and both the forward and backward passes must use all-reduce
to combine the sharded results. This is why "TP is only done inside a node's NVLink domain, and saturates as soon as $W$ is small." This problem computes the per-layer communication bytes of TP
(pure computation, with a custom simplified model).

## Model

Megatron style: the MLP and attention of a transformer layer each use a "column-parallel → row-parallel" combination, which means
- **Forward**: attention 1 all-reduce + MLP 1 all-reduce $=2$ times;
- **Backward**: another $2$ corresponding times.

That is **4 all-reduces per layer** in total, and each message is one **activation** tensor:
$$ \text{act\_bytes} = \texttt{batch}\times \texttt{seq}\times \texttt{hidden}\times \texttt{bytes\_per\_elem} $$

With $W$ GPUs in a ring, one all-reduce sends and receives $2c\cdot \text{act\_bytes}$ per GPU (ring coefficient $c=\frac{W-1}{W}$, the same coefficient as in
`parallel-allreduce` and `distributed-zero-communication`). Hence the communication volume per layer per GPU:
$$ \text{per\_layer} = 4\times\big(2c\cdot \text{act\_bytes}\big) = 8\,\frac{W-1}{W}\cdot \text{act\_bytes} $$
$$ \text{total} = \texttt{n\_layers}\times \text{per\_layer} $$

## Rules

Implement `solve(batch, seq, hidden, world, n_layers=1, bytes_per_elem=2)`, returning the triple
`(act_bytes, per_layer_comm, total_comm)` (float byte counts). When $W=1$, $c=0$ and the latter two terms are `0.0` (no TP communication on a single GPU).

## Examples

```
batch=8, seq=2048, hidden=4096, world=8, n_layers=32, bytes_per_elem=2
act_bytes    = 8·2048·4096·2      = 1.34217728e8
c            = 7/8 = 0.875
per_layer    = 8·0.875·1.342e8    = 9.39524096e8
total        = 32·9.395e8         = 3.0064771072e10
→ (1.34217728e8, 9.39524096e8, 3.0064771072e10)
```

## Constraints

- `batch, seq, hidden ≥ 1`; `world ≥ 1`; `n_layers ≥ 0`; `bytes_per_elem` is typically 2 (fp16/bf16) or 4
- Return `(float, float, float)`, compared with relative error $10^{-6}$
- The Megatron simplification of "4 all-reduces per layer" is fixed: it does not model the all-gather/reduce-scatter
  replacement introduced by Sequence Parallelism, nor does it consider communication/computation overlap
