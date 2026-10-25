# ZeRO Communication Volume: DDP vs Stage 1/2 vs Stage 3

## Problem

`training-state-memory` computes **how much GPU memory** each ZeRO stage saves; this problem computes its cost—**how many bytes each GPU must
communicate per iteration**. The more thoroughly memory is partitioned, the more parameters have to be moved around. This is a pure computation problem (custom simplified formulas, no framework execution).

## Model and Notation

$W$ ranks form a ring. Ring-based primitives such as All-Reduce / Reduce-Scatter / All-Gather all carry the same
ring coefficient in their **per-GPU communication volume**:
$$ c = \frac{W-1}{W} $$
(A single All-Reduce over data of size $S$ sends and receives $2cS$ per GPU in total; a Reduce-Scatter alone or an All-Gather alone
costs $cS$.) Let the parameter count be $N$, the **gradient bytes** to synchronize per iteration $G = N \cdot \texttt{grad\_bytes}$, and
the **parameter bytes** $P = N \cdot \texttt{param\_bytes}$.

Per-GPU communication volume of each scheme (simplified teaching model, in integer multiples):

- **DDP**: one All-Reduce over the gradients $= 2cG$. Parameters and optimizer states stay local, with no extra movement.
- **ZeRO Stage 1 / 2**: gradients use Reduce-Scatter ($cG$, each rank reduces only the shard it owns) $+$ All-Gather
  the updated fp16 parameters back ($cG$) $= 2cG$. **Same as DDP**—what is saved is memory, not communication.
- **ZeRO Stage 3**: beyond the $2cG$ above, the parameters are also partitioned, so forward and backward each need a temporary All-Gather of the parameters
  (each $cP$) $= 2cG + 2cP$. **More communication**, trading bandwidth for memory.

## Rules

Implement `solve(n_params, world, grad_bytes=2, param_bytes=2)`, returning the triple
`(ddp, zero_1_2, zero_3)`—the **per-GPU, per-iteration communication bytes** of the three schemes (float).
When $W=1$, $c=0$, and all three should be `0.0` (no communication on a single GPU).

## Examples

```
N=1e9, W=8, grad_bytes=2, param_bytes=2
G = P = 2e9,  c = 7/8 = 0.875
ddp     = 2·0.875·2e9 = 3.5e9
zero_1_2= 2·0.875·2e9 = 3.5e9
zero_3  = 3.5e9 + 2·0.875·2e9 = 7.0e9
→ (3.5e9, 3.5e9, 7.0e9)
```

## Constraints

- `1 ≤ world ≤ 4096`; `n_params ≥ 1`; `grad_bytes` and `param_bytes` are typically 2 (fp16/bf16) or 4 (fp32)
- Return `(float, float, float)`, compared within relative error $10^{-6}$
- Compute only with the simplified model defined above; no real bandwidth/overlap is involved
