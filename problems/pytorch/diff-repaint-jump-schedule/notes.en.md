# Solution Notes · RePaint Jump Schedule

## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_repaint.py`: `RePaintScheduler.set_timesteps` (lines 188–231) is exactly what this problem reproduces. The masked blending, variance and reused noise inside `step` are a different story (several comments there point at Algorithm 1 of the RePaint paper, arXiv:2201.09865, visible at lines 340–390); this problem does not touch them.
- The intuition behind RePaint: a diffusion model generates unconditionally but does not guarantee that a filled region matches the known region along the seam, so it periodically jumps back to a noisier level and regenerates that stretch — effectively repeating a "noise up, denoise down" harmonization.

## Core Points and Pitfalls

**After a jog the loop keeps descending from the high point, not from the original one.** The easiest mistake: assuming it returns to `t` after walking `j` steps up. In the source, `t` is raised in place and the `while` loop continues from there, so every intermediate level is **walked again**, and jump budgets on those levels can fire again. The 26 entries (not 10 or 18) for `T=100, N=10, j=2, n=2` follow directly from this.

**Jump points are `range(0, N - j, j)`.** The upper bound \(N-j\) is **exclusive**: if a point \(t\) satisfies \(t+j\ge N\), jogging up would exceed the maximum timestep \(N-1\), so the source never puts it in the dict. Writing `range(0, N, j)` adds out-of-range entries at the tail.

**The budget is `n-1`, not `n`.** The first descent to that point consumes one jog, so `n=1` means "never jump". Thus `n` is "at most n consecutive jogs starting here", not "jog n times".

**Two integer divisions.** The loop bound is `min(T, N)` (inference steps cannot exceed training steps) and the step ratio is `T // N_eff`. Both use the clamped \(N_{\text{eff}}\): with `T=8, N=20` the ratio is 1, not 0, and the output is `7..0` with 8 entries.

**The final `* step` is a vectorized multiply, not an index.** The loop works entirely in \([0, N_{\text{eff}}-1]\) and scales once at the end; scaling inside the loop desynchronizes the `jumps` keys.

**Structural checks catch more than value comparison.** Adjacent differences must be exactly \(\pm\)step, every ascending run must have length \(j\), and the number of runs must equal \((n-1)\times\)(number of jump points) — together these cover "jumped the wrong way", "missed a jog" and "did not re-walk after a jog". Note you must **not** expect the skeleton (sequence with ascents removed) to be monotone: the re-walk after a jog leaves a new higher value, so the skeleton jumps up at the seam.

**Return int64.** Timesteps are integer indices in the scheduler; a float tensor makes the subsequent `alphas_cumprod[t]` fail outright.

## Division of Labor

- `diff-ddim-inverse-step`: also "move back to a noisier level", but that is the continuous arithmetic of one step; this problem is **discrete calendar design** with no alpha arithmetic.
- `diff-sdedit-strength`: decides at which level inversion starts (truncation); this problem decides when each level of the loop is repeated.
- `diff-ddpm-step` / `diff-ddim`: the per-step numeric update; this problem produces the timestep sequence that drives them.
- `diff-aspect-ratio-buckets`: another "build a discrete table, then assign" scheduling problem, but that table is over resolutions rather than time.
