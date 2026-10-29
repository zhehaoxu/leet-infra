# RePaint Jump Schedule: Timesteps That Travel Backwards in Time

## Problem Statement

RePaint fills a masked region using information from the known region: instead of denoising monotonically from 0 to T, it periodically winds the timestep **back** (toward more noise), so the model regenerates and gradually aligns the statistics on both sides of the seam. That "long jog forward, short jog back" calendar is controlled by `jump_length` (how many steps each backward jog takes) and `jump_n_sample` (how many jogs may start at each jump point).

This problem computes **the calendar itself**: given the training steps, the inference steps and the two jump parameters, produce the full 1-D integer sequence that the scheduler executes item by item.

## Rules

Let \(N_{\text{eff}}=\min(T,N)\) be the number of inference steps actually used.

1. **Jump points**: among \(0, j, 2j, \dots\), take every point strictly below \(N_{\text{eff}}-j\); each starts with a budget of \(n-1\) (\(j\) = `jump_length`, \(n\) = `jump_n_sample`).
2. **Main loop** (starting from \(t=N_{\text{eff}}\)):

   ```
   while t >= 1:
       t = t - 1
       emit t
       if budget[t] > 0:
           budget[t] -= 1
           repeat j times:
               t = t + 1
               emit t
   ```
   The catch is that after a jog the loop **does not** resume below the old position: it keeps decreasing from the raised \(t\), so the \(j\) levels just crossed are walked again — and so is everything below them. That is why the sequence is much longer than \(N_{\text{eff}}\).
3. **Convert to training steps**: multiply the whole sequence by the step ratio \(\lfloor T/N_{\text{eff}}\rfloor\) (integer division).

Return a 1-D `int64` tensor.

## Example

`T = 100`, `N = 10`, `j = 2`, `n = 2`: the jump points are 0, 2, 4, 6, with one jog each. On reaching 6 the budget is 1, so it jogs up to 8 and descends again from there; it jogs once more at 4, 2 and 0.

```
[90, 80, 70, 60, 70, 80, 70, 60, 50, 40, 50, 60, 50, 40,
 30, 20, 30, 40, 30, 20, 10, 0, 10, 20, 10, 0]
```

26 entries (step ratio 10). The `60 → 70 → 80` stretch is an ascent of exactly \(j=2\) steps, and the `70, 60, 50, 40` that follows walks everything below 6 again.

With `T = 30`, `N = 10`, `j = 3`, `n = 2` (jump points 0, 3, 6, step ratio 3):

```
[27, 24, 21, 18, 21, 24, 27, 24, 21, 18, 15, 12, 9,
 12, 15, 18, 15, 12, 9, 6, 3, 0, 3, 6, 9, 6, 3, 0]
```

28 entries.

## Constraints

- All four parameters must be Python integers \(\ge 1\) (booleans do not count and raise `ValueError`);
- When `num_inference_steps > T`, clamp to \(N_{\text{eff}}=T\) instead of raising; the step ratio uses \(N_{\text{eff}}\);
- When `jump_length` is at least \(N_{\text{eff}}\) the jump-point set is empty and the output degenerates to the arithmetic descent \(N_{\text{eff}}-1, N_{\text{eff}}-2, \dots, 0\);
- When `jump_n_sample = 1` every budget is 0, so again there are no jogs;
- Properties: the first entry is \((N_{\text{eff}}-1)\cdot\lfloor T/N_{\text{eff}}\rfloor\) and the last is 0; every adjacent difference is exactly \(\pm\lfloor T/N_{\text{eff}}\rfloor\); each maximal ascending run has length exactly \(j\), and the number of such runs equals \((n-1)\times\) the number of jump points.
