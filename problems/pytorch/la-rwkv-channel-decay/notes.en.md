# notes — la-rwkv-channel-decay (order 658)

## Source

- Official implementation: `BlinkDL/RWKV-LM`, `RWKV-v5/src/model.py`, class `RWKV_Tmix_x060` (`--my_testing x060` enables v6.0); core op `WKV_6` (`cuda/wkv6_cuda.cu`). `BlinkDL/ChatRWKV`'s `RWKV_in_150_lines.py` is the v4 formulation (constant `time_decay` added to pp) — **do not confuse**; the `BlinkDL/RWKV-LM/RWKV-v6/` folder holds only a README, actual v6 training code lives in the RWKV-v5 folder behind the x060 flag.
- Model side: `w = time_decay + tanh(xw @ time_decay_w1) @ time_decay_w2` (data-dependent, negative), `u = time_faaaa` (per-channel parameter); GroupNorm inside heads, token-shift lerp, silu gating all live in `RWKV_Tmix_x060` and are **out of scope** — this problem only covers the WKV core recurrence ledger.
- The official CUDA kernel's per-channel loop is exactly the two branches above; float32 internally, single sequence (no batch dim); initial `aa=bb=0`, `pp` a large negative proxy for -inf. This problem pins that proxy constant at `-1e38` (representable in float32).

## Judging surface

- solve args: r/k/v/w `[T,C]`, u `[C]`, state dict (a/b/p each `[C]`, may be None). Returns `wkv`/`y`/`final_state`/`p_trace` (nested lists, per-token exponent bound making the decay arithmetic observable).
- Oracle: `test_cases._py`, an independent pure-Python implementation (math.exp per channel), comparing y/wkv/final state/p_trace at 1e-5 tolerance.
- Three hand-computed anchors: (1) zero initial state gives `wkv=v` at t=0 and `p_trace[0]=k0` (numerical absorption of -1e38); (2) initial `(a,b,p)=(1,1,-2)`, w=-1, k=-3: output branch `p_out=max(-2,0)=0`, `wkv=(e^-2+2)/(e^-2+1)`, update branch `ww=-3=p`, final a=3, b=2, then t1 verifies the `e^-1` decay; (3) bonus branch analytic form with `u=0.7 > pp=0`.
- Causality case: full run equals two-chunk run (second chunk fed the final state) — bitwise identical.
- Channel-independence case: perturbing only channel 1 leaves channels 0/2 bitwise unchanged.
- Invalid inputs: shape mismatches among the four tensors, u channel mismatch, empty T/C, missing state keys / wrong lengths → ValueError.

## Mutation log (10)

9 valid mutations all caught: bonus-to-zero, decay-on-new-p (`ww = p_new + w`, the classic v6 transcription bug), no-decay, no-exp-numer / no-exp-denom, out-no-e2 / out-no-denom-e2, no-gate, state-reset, p-init-zero (caught by the zero-state case at t≥1).

**One equivalent mutation (verified, not a test gap)**: `bonus-to-k` — changing the output branch's `p_out = max(pp, u)` to `max(pp, k)`. Reason: `wkv = (e1·aa + e2·v)/(e1·bb + e2)` has e1/e2 sharing the same p_out; multiplying numerator and denominator by `exp(p_out)` gives `wkv = (exp(pp)·aa + exp(u)·v)/(exp(pp)·bb + exp(u))`, **independent of p_out** — the max shift is pure numerical stabilization and its comparison operand is behaviorally free. But the `u` inside `e2 = exp(u - p_out)` must survive (bonus-to-zero catches dropping it).

## Edge cases and pitfalls

- `pp + w` is numerically absorbed when pp=-1e38 (both float32/float64), so the first frame's decay is invisible from a zero state — anchors must provide a finite initial p to observe w.
- y is computed as `rt * wkv_t` inside the loop and re-stacked on return; the two must agree.
- Red lines: model-side token-shift lerp / LoRA-generated w / GroupNorm / head structure are out; no batch dim (single-sequence kernel convention).
