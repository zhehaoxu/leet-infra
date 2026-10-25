# Training-State Memory: the Four Copies and ZeRO Sharding

## Problem

During mixed-precision + Adam training, besides activations, every parameter is split into several copies
stored in GPU memory. This is exactly where "why does a 7B model need tens of GB of memory" comes from.
This problem tallies the **bytes of optimizer state per parameter**, and computes the **per-GPU memory
after each of the three ZeRO sharding stages**.

## Rules

- Implement `solve(n_params, world)`, returning the 4-tuple `(stage0, stage1, stage2, stage3)`, all in
  **bytes per GPU** (float).
- Use the standard mixed-precision + Adam **bytes-per-parameter** accounting (fp16 stores weights/gradients,
  fp32 stores the master weights and optimizer moments):

  | Component | Precision | Bytes/param |
  |------|------|-----------|
  | fp16 parameters (used in forward) | fp16 | 2 |
  | fp16 gradients | fp16 | 2 |
  | fp32 master weights | fp32 | 4 |
  | Adam first moment m | fp32 | 4 |
  | Adam second moment v | fp32 | 4 |
  | **Total** | | **16** |

- Let $N$ = `n_params` and $W$ = `world` (the number of GPUs in data parallelism). ZeRO **evenly splits**
  these components across the $W$ GPUs:
  - **Stage 0 (no sharding)**: every GPU holds the full set → $16N$.
  - **Stage 1**: only the **optimizer state** (master+m+v = 12 bytes) is sharded; parameters + gradients are
    still replicated → $N\left(4 + \frac{12}{W}\right)$.
  - **Stage 2**: the **gradients** (2 bytes) are sharded as well, parameters are still replicated →
    $N\left(2 + \frac{14}{W}\right)$.
  - **Stage 3**: parameters / gradients / optimizer state are **all sharded** → $\frac{16N}{W}$.
- **Activation memory is not counted here** (covered by a separate problem); only weight-related training
  state is included.

## Example

```
Input: n_params=1_000_000_000 (1B), world=8
  stage0 = 16*1e9            = 16,000,000,000
  stage1 = 1e9*(4+12/8)=5.5e9 = 5,500,000,000
  stage2 = 1e9*(2+14/8)=3.75e9= 3,750,000,000
  stage3 = 16e9/8             = 2,000,000,000
Output: (16000000000.0, 5500000000.0, 3750000000.0, 2000000000.0)
  (roughly 14.90 / 5.12 / 3.49 / 1.86 GiB per GPU)
```

## Constraints

- `n_params` is a positive integer, `world` is a positive integer ≥1
- Return a **tuple** `(stage0, stage1, stage2, stage3)` whose elements are floats (bytes)
- The judge compares each element with a relative tolerance (`1e-6`)
