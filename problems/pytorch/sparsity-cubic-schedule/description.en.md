# Sparsity: Cubic Gradual Sparsity Schedule (GSL)

## Problem

Pruning the network to the target sparsity in one shot hurts training stability. The **Gradual Sparsity / GSL** schedule, proposed by Zhu & Gupta (*To Prune, or Not to Prune*, arXiv:1710.01878), uses a **cubic function** to smoothly push the sparsity from an initial value \(p_i\) to a final value \(p_f\) within a training window, pruning hardest early and stabilizing later:

\[ p_t = p_f + (p_i - p_f)\left(1 - \frac{t - t_0}{t_1 - t_0}\right)^3 \]

where \(t\) is the current step and \([t_0,t_1]\) is the pruning window. Before the window the sparsity stays at \(p_i\); after the window it is constantly \(p_f\).

## Rules

Implement `solve(initial_sparsity, final_sparsity, current_step, first_step, final_step)`, returning the target sparsity \(p_t\) at the current step (a 0-dim float32 scalar):
- If \(\text{current\_step} \le t_0\): return \(p_i\);
- If \(\text{current\_step} \ge t_1\): return \(p_f\);
- Otherwise apply the cubic interpolation above.

Ready-made scheduling libraries are forbidden.

## Examples

`initial_sparsity=0.0`, `final_sparsity=0.8`, `first_step=0`, `final_step=1000`:

```
current_step=500 → frac=0.5, (1-0.5)^3=0.125, p=0.8+(0-0.8)*0.125 = 0.7
current_step=0   → p_i = 0.0
current_step=1000→ p_f = 0.8
```

## Constraints

- CPU float32; \(t_1 > t_0\).
- The cubic term decays monotonically as \(t\) goes from \(t_0\to t_1\), so the sparsity approaches \(p_f\) monotonically.
- Closed-interval boundaries: the two window endpoints take \(p_i\) and \(p_f\) respectively.
