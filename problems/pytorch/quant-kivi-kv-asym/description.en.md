# Quantization: KIVI Asymmetric KV Cache Quantization (Per-Channel Keys, Per-Token Values)

## Problem

At long context, the KV cache outgrows the weights. **KIVI** (Liu et al., ICML 2024, arXiv:2402.02750) observes that keys and values have different distribution structure, and gives them **different quantization granularities**:

- **Keys, per-channel**: all tokens share one set of quantization parameters per feature channel. The channel axis (`head_dim`) is the "stable direction" of the key distribution — a channel's variance is large while its mean is stable across tokens, so channel-wise statistics are the right choice; quantizing keys token-by-token only amplifies noise.
- **Values, per-token**: every token shares one set of parameters across channels. Value distributions shift sharply from token to token, so per-token quantization keeps up.

Both use **asymmetric** quantization (non-zero zero-point): weights and activations are often offset wholesale, and symmetric quantization would waste half the code space on an empty range. The convention used here (matching KIVI) is

\[
\Delta = \frac{x_{\max}-x_{\min}}{2^{b}-1}\quad(\text{if }x_{\max}=x_{\min},\ \Delta=1),\qquad
q=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x-x_{\min}}{\Delta}\right),0,2^b-1\right),\qquad
\hat x=q\,\Delta+x_{\min}
\]

i.e. the zero-point **is** `x_min` itself (floating point, never rounded, never clamped) — `x_min` maps exactly to code 0 and `x_max` to code `2^b-1`.

## Task

Implement `solve(K, V, bits)`:

- Input: `K`, `V` are float32 `(T, D)` (token axis first, channel axis last), `bits` an integer in 2..8.
- For **K**, reduce over the token axis (`dim=0`) for min/max: one set of parameters per channel, length `D`;
- For **V**, reduce over the channel axis (`dim=1`): one set per token, length `T`;
- Return `(K_deq, V_deq, K_scale, K_zero, V_scale, V_zero)`: dequantized results are float32 `(T, D)`; `*_scale` and `*_zero` are float32 with lengths `D`, `D`, `T`, `T`.

**Conventions**: compute internally in float64, output float32; tolerance \(10^{-5}\). Invalid `bits` (`<2` or `>8`) raises `ValueError`.

## Example

```
K = [[0, 1], [2, 3]], V = [[0, 1], [2, 3]], bits = 4
K per-channel: ch0 [0,2] → Δ=2/15, min=0 → K_deq[:,0]=[0,2]
               ch1 [1,3] → Δ=2/15, min=1 → K_deq[:,1]=[1,3]
V per-token:   t0 [0,1] → Δ=1/15, min=0 → V_deq[0]=[0,1]
               t1 [2,3] → Δ=1/15, min=2 → V_deq[1]=[2,3]
```

(Every channel/token min and max lands exactly on code 0 and code 15, so the round trip is lossless.)

## Constraints

- The reduction axis is the first judged point: K along `dim=0`, V along `dim=1`; swapping them is caught immediately by the granularity assertions.
- Degenerate groups (`max == min`): `Δ = 1`, all codes 0, reconstruction exactly equals the input.
- `round` is round-half-to-even (same as `torch.round`).
