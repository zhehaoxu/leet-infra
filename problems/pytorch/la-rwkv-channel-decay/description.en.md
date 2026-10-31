# RWKV v6 Channel-wise Decay: WKV Recurrence with Per-token w/u Ledger

## Problem

RWKV v6 (x060) replaces the constant `time_decay` of v4/v5 with a **data-dependent, per-token per-channel decay**, and the bonus (`time_faaaa`) becomes per-channel as well. This problem transcribes the official `wkv6` CUDA kernel as a CPU ledger: given receptance `r`, key `k`, value `v` (each `[T, C]`), the per-token decay `w` (`[T, C]`), the per-channel bonus `u` (`[C]`) and an optional initial state `(a, b, p)` (each `[C]`), run the per-channel log-space state triplet `(aa, bb, pp)` token by token, and return the outputs before/after the receptance gate plus the final state.

## Solution

Each channel independently maintains the numerically stable triplet (numerator `aa`, denominator `bb`, exponent bound `pp`), all in float32:

**Output branch** (bonus compared against the old exponent):

```
p_out = max(pp, u)
e1 = exp(pp - p_out);  e2 = exp(u - p_out)
wkv = (e1·aa + e2·v) / (e1·bb + e2)
```

**Update branch** (decay applied to the **old** exponent pp, then compared with the current k):

```
ww = pp + w
p   = max(ww, k)
e1  = exp(ww - p);  e2 = exp(k - p)
aa  = e1·aa + e2·v
bb  = e1·bb + e2
pp  = p
```

Final output `y = r * wkv` (receptance multiplied back as-is; the sigmoid lives in the model, not here).

Key structural points:

1. **w applies to the old exponent**: in `ww = pp + w`, `pp` is the old exponent used by this token's output branch, not the `p_out` it produced. This is the single most common transcription bug.
2. **The bonus only enters the output branch**: `u` participates in the "read old state + current v" interpolation; the update branch never touches `u`.
3. **Initial state**: by default `aa = bb = 0`, `pp = -1e38` (the official kernel's -inf proxy). With a zero state, `wkv = v` at t=0 always holds; `pp + w` is numerically absorbed into `-1e38`, so `p_trace[0] = k0`.
4. **Channel independence**: the C channels never interact; `p_trace` records the exponent bound per token so the decay arithmetic is observable.
5. Feeding a sequence in two chunks equals feeding it whole (causality of the recurrence), and the final state supports resumption.

## Constraints

- 1 ≤ T ≤ 32, 1 ≤ C ≤ 16; all tensors are float32.
- Shape mismatches, an empty sequence, or an incomplete state dict → `ValueError`.
