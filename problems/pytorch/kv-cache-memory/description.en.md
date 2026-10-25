# KV Cache Memory Estimation

## Problem

Autoregressive decoding caches the already-computed Key/Value of every layer (the KV cache) to avoid recomputation. During long-context inference,
**the KV cache often takes more GPU memory than the model weights themselves**. This problem computes the number of bytes occupied by the KV cache for a given
configuration, and shows how MQA / GQA reduce this memory by sharing KV heads.

## Rules

- Implement `solve(layers, heads, head_dim, seq, batch, dtype_bytes=2, kv_heads=None)`, returning the KV cache size in bytes (an integer).
- The cache stores **both K and V**, and its size grows with layers × KV heads × per-head dim × sequence length × batch × bytes per element:
  $$ \text{bytes} = \underbrace{2}_{K,V}\cdot \text{layers}\cdot \text{kv\_heads}\cdot \text{head\_dim}\cdot \text{seq}\cdot \text{batch}\cdot \text{dtype\_bytes} $$
- `kv_heads` determines how many groups of K/V are stored in the cache:
  - `None` (default, MHA): each query head has its own KV, `kv_heads = heads`;
  - `kv_heads=1` (MQA): all query heads share one set of KV, and the cache shrinks to $1/\text{heads}$;
  - `1 < \text{kv\_heads} < \text{heads}` (GQA): grouped sharing, shrinking to $\text{heads}/\text{kv\_heads}$.
- `seq` is the **number of cached tokens** (generation length), not an increment beyond the prompt.
- `head_dim` is given directly (it is not required to equal `hidden/heads`).

## Examples

```
Input: layers=32, heads=32, head_dim=128, seq=2048, batch=1, dtype_bytes=2
  MHA (kv_heads=None):  2*32*32*128*2048*1*2 = 1,073,741,824  (1 GiB)
  MQA (kv_heads=1):     2*32*1 *128*2048*1*2 =    33,554,432  (32 MiB)
  GQA (kv_heads=8):     2*32*8 *128*2048*1*2 =   268,435,456  (256 MiB)
solve(32,32,128,2048,1,2)         == 1073741824
solve(32,32,128,2048,1,2, kv_heads=1) == 33554432
```

## Constraints

- `layers, heads, head_dim, batch` are positive integers; `seq ≥ 0`
- `dtype_bytes` ∈ {1, 2, 4}; `kv_heads` is `None` or an integer with $1\le \text{kv\_heads}\le \text{heads}$
- Return the size in bytes as an **integer** (Python int)
