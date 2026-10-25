# Collective Communication Latency Accounting: Ring vs Recursive-Doubling

## Problem

The same All-Reduce can be implemented over completely different communication topologies: **Ring** takes many steps but moves only a small amount of data per step and has optimal bandwidth utilization; **Recursive-Doubling (binary/butterfly)** needs only $\log_2 W$ steps and has low latency, but moves more data per step. Which is faster depends on "message size vs per-call startup overhead" — exactly the question the classic **α-β cost model** answers (NCCL's automatic algorithm selection is computing this trade-off).

## α-β Model (use exactly these formulas, no other variants)

Notation: $W$ = number of ranks ($W$ is a power of 2), $S$ = message bytes per rank, $\alpha$ = startup/latency overhead per step (seconds), $\beta$ = transmission time per byte (seconds/byte).

- **Ring All-Reduce**: $2(W-1)$ steps, moving $\frac{S}{W}$ bytes per step:
  $$ T_{ring} = 2(W-1)\left(\alpha + \frac{S}{W}\,\beta\right) $$
- **Recursive-Doubling All-Reduce**: $\log_2 W$ steps, moving the entire $S$ bytes per step:
  $$ T_{rd} = \log_2 W\,(\alpha + S\,\beta) $$

Intuition: for large $S$ → ring's bandwidth term $\frac{2(W-1)}{W}S\beta\ (\approx 2S\beta)$ is far smaller than rd's $\log_2 W\cdot S\beta$, so ring wins; for large $\alpha$ and small $S$ → rd wins with fewer steps ($\log_2 W \ll 2(W-1)$).

## Rules

Implement `solve(W, S, alpha, beta)` and return a dictionary:

- `"ring_time"`: $T_{ring}$ (float)
- `"rd_time"`: $T_{rd}$ (float)
- `"faster"`: `"ring"` if $T_{ring}\le T_{rd}$, otherwise `"recursive_doubling"` (**ties go to ring**)
- `"speedup"`: slower / faster $=\max/\min$ (float, $\ge 1$)

## Examples

```
W=8, S=1e6, alpha=1e-5, beta=1e-9:
  T_ring = 14*(1e-5 + 125000*1e-9) = 0.00189
  T_rd   = 3*(1e-5 + 1e-3)         = 0.00303
  → {"ring_time":0.00189, "rd_time":0.00303, "faster":"ring", "speedup":1.6031746}
```

## Constraints

- $W\in\{2,4,\dots,64\}$ and a power of 2; $S,\alpha,\beta > 0$ (float)
- Use strictly the two formulas above; floating point is compared with relative error $10^{-6}$, `faster` with exact string matching
- No concrete bandwidth/topology simulation is involved; this is pure closed-form algebra
