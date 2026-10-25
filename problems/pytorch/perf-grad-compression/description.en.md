# Gradient Compression Bandwidth: One-Bit Quantization vs Low-Rank Projections

## Problem

Data parallelism does one all-reduce per step, so **the communication volume is proportional to the bytes of the
gradients**; across nodes the NIC (tens of GB/s) is often an order of magnitude slower than NVLink, which makes
"send fewer bytes" the most direct optimisation. **1-bit Adam / sign compression** squashes every element to 1 bit
(plus one fp32 scale shared by a group of elements), while **PowerSGD** rewrites each matrix as two low-rank factors
$P Q^{\top}$ and only ships them. Both rely on **error feedback** - a local fp32 residual accumulator that carries
the dropped mass to the next step - so they **save bandwidth but not memory**. Compression is not free either: once
the bit width grows, the groups get finer, or the rank nears full rank, the compressed payload **exceeds the plain
fp32 one** and the implementation has to fall back to the uncompressed all-reduce. This problem turns that decision
into a pure byte ledger.

`distributed-zero-communication` accounts for communication on **uncompressed** gradients, `operator-fusion` only
counts kernel HBM traffic, `transformer-mem` only counts bytes of parameters; this problem is only about the
**bytes actually moved after compression, the fraction saved, and the two switch points (fallback and maximum
feasible bit width)**.

## Rules

- **Payload** (integer ceiling; matrices with $numels=0$ are skipped entirely): `full` is total elements times bytes
  per element; `sign` per matrix is "sign bits packed, $\lceil numels\cdot bits/8\rceil$ bytes" plus
  "group scales $\lceil numels/g\rceil\cdot 4$ bytes", where $g=0$ means one scale for the whole tensor (4 bytes);
  `lowrank` ships $(rows+cols)\cdot r$ elements per matrix, but when $r\ge\min(rows,cols)$ the matrix is treated as
  **full rank** and shipped as is.
- **Fallback**: `compressed = min(payload, full_payload)` and `fallback = payload > full_payload` (equality does not
  trigger a fallback).
- **Ring coefficient** reused from `distributed-zero-communication`: $c=(W-1)/W$, one all-reduce moves $2cS$ bytes
  per rank, so with $W=1$ both communication figures are $0$. `saved_ratio` is a ratio of payloads and does not
  depend on $W$.
- **`max_bits`** is only defined for `method="sign"`: scan the bit width from 1 to `bytes_per_element*8` and keep the
  **largest** width that still satisfies `payload <= full_payload` (the payload is monotone non-decreasing in the
  bit width; if none fits, report 0, and 0 also when `full_payload == 0`). `full` and `lowrank` always report 0.
- The exact meaning of every input, each formula and the return-tuple typing live in the `starter.py` docstring,
  which is the single source of truth.

## Examples

```
Input: shapes=[[4096,4096]], world=8, method="sign", bytes_per_element=4, bits_per_element=1
  full_payload = 4096*4096*4 = 67,108,864
  payload = ceil(16,777,216*1/8) + 4 = 2,097,152 + 4 = 2,097,156 (g=0: one scale for the tensor)
  c = 7/8 -> per_gpu = 2c*2,097,156 = 3,670,023.0    full_comm = 2c*67,108,864 = 117,440,512.0
  saved = 1 - 2,097,156/67,108,864 = 0.9687499403953552   ef = 16,777,216*4 = 67,108,864
  max_bits = 31 (at 32 bits, 2,097,152*32 + 4 already exceeds 67,108,864)
Output: (67108864, 2097156, 3670023.0, 117440512.0, 0.9687499403953552, 67108864, False, 31)

Input: shapes=[[1280,3584]], world=8, method="lowrank", bytes_per_element=2, rank=8
  full_payload = 1280*3584*2 = 9,175,040   payload = (1280+3584)*8*2 = 77,824
  per_gpu = 1.75*77,824 = 136,192.0        full_comm = 1.75*9,175,040 = 16,056,320.0
  saved = 1 - 77,824/9,175,040 = 0.9915178571428571   ef = 4,587,520*4 = 18,350,080
Output: (9175040, 77824, 136192.0, 16056320.0, 0.9915178571428571, 18350080, False, 0)

Input: shapes=[[1024,1024]], world=4, method="sign", bytes_per_element=1, bits_per_element=8,
       scale_group_size=128
  full_payload = 1,048,576 (an fp8 baseline)   payload = 1,048,576 + ceil(1,048,576/128)*4 = 1,081,344
  payload > full -> fallback=True, compressed=1,048,576, saved=0.0, both comm figures 1,572,864.0
  max_bits = 7 (8 bits already ties the baseline, and the scale then pushes it over)
Output: (1048576, 1048576, 1572864.0, 1572864.0, 0.0, 4194304, True, 7)
```

## Constraints

- `shapes` has length $\ge 1$, each entry is $[rows, cols]$ (each $\ge 0$; a $0$ means that block does not communicate);
  `world` $\ge 1$; `method` $\in$ `{"full","sign","lowrank"}`; `bytes_per_element` $\ge 1$;
  `bits_per_element` $\ge 1$; `scale_group_size` $\ge 0$; `rank` $\ge 1$
- Return the 8-tuple `(full_payload, compressed_payload, per_gpu_comm_bytes, full_comm_bytes, saved_ratio,
  ef_state_bytes, fallback, max_bits)`: items 1, 2, 6, 8 are ints (exact, `torch.equal` semantics), items 3-5 are
  floats (`atol=1e-5, rtol=1e-5`), item 7 is a bool
- A pure byte ledger: no convergence-loss model, no top-k sparsification, no shard-ownership simulation
