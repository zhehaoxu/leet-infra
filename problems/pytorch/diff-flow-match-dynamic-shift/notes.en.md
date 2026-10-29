# Deep Dive: Flow Match Dynamic Time Shifting and Terminal Stretching

## Where it comes from

`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`: `time_shift` (dispatching to `_time_shift_exponential` / `_time_shift_linear`), `stretch_shift_to_terminal`, and the ordering of steps 2–6 inside `set_timesteps` (shift → stretch → converters → multiply by `num_train_timesteps` → append the terminal value). `shift_terminal` originates from LTX-Video's `ltx_video/schedulers/rf.py`, linked in the upstream comment. In Flux-style pipelines the dynamic \(\mu\) is derived from the resolution (`base_shift` / `max_shift` / `base_image_seq_len` / `max_image_seq_len`); that part is already covered by `diff-rf-timestep-shift`, so this problem starts from a known \(\mu\).

## Key ideas and pitfalls

1. **The two formulas are the same family**: the static shift \(s\cdot x/(1+(s-1)x)\) and the linear shift \(\mu x/(1+(\mu-1)x)\) coincide when \(\mu=s\), and the exponential shift matches them when \(\mu=\ln s\). Upstream writes three because Flux interpolates \(\mu\) from resolution, Dev uses a static shift and SD3.5 uses the exponential type. An implementation can get away with `e^\mu/(e^\mu+x)` and `\mu/(\mu+x)`, but **the exponential branch is scaled by \(e^\mu\), not by \(\mu\)** — treating both as the same `mu` is the most common formula error here.
2. **The endpoints are pinned**: at \(x=1\), \((1/x-1)=0\), so every shift maps to 1; at \(x=0\), \(1/x=\infty\) and the shift maps to 0. With `sigma_max=1, sigma_min=0` the first and last sigmas never move, only the middle does. Implementing via `1/x` divides by zero at \(x=0\): torch yields `inf` and naturally produces 0, while a pure-Python oracle must short-circuit explicitly.
3. **The ordering inside `stretch_shift_to_terminal`**: `scale = (1 - z[-1]) / (1 - shift_terminal)` must use the **post-shift** last entry. Stretch-then-shift would let the shift move the last entry again, so `shift_terminal` would miss. The tests build the wrong-order result explicitly and require its last entry to differ from the correct one by more than 1e-3.
4. **The timesteps use the stretched sigmas**: `timesteps = sigmas * T` is computed after stretching, so `timesteps[-1] == shift_terminal * T` rather than the pre-stretch `0`.
5. **`invert_sigmas` recomputes twice**: it sets `sigmas = 1 - sigmas` and then `timesteps = sigmas * T` (it does not negate or reuse the old timesteps), and it appends **1** instead of 0. After inversion the sigmas increase, starting at 0 and (with a terminal set) ending at `1 - shift_terminal`.
6. **\(N=1\) cannot coexist with a terminal**: a single sigma makes `1 - z[-1]` zero, so `scale = 0/(1-terminal)` and `(1-z)/scale` is `0/0` → NaN. Upstream does not guard this; the problem rejects it in validation and says so in the statement.
7. **The linear shift needs \(\mu>0\)**: with \(\mu=0\) the numerator is identically zero and every sigma collapses to 0, whereas the exponential branch with \(\mu=0\) is the identity (since \(e^0/(e^0+x)\) keeps its shape). The two behave asymmetrically at \(\mu=0\), hence the lower bound applies only to the linear branch.
8. **The terminal value may duplicate the last entry**: with `sigma_min=0` the last entry is already 0, and the appended terminal is 0 too, so `sigmas` ends in two zeros. That is the normal upstream shape, not a missing write.
9. **Inversion is not an involution**: `invert_sigmas` applies before the terminal append, so "invert twice" does not return the original sequence (the terminal goes 0 → 1 and back, but the intermediate entries and the timestep derivation already took a different path). The tests cross-check element-wise instead of asserting involution.

## How the judge cross-checks

The judge uses a pure-Python scalar oracle (`_oracle`) that redoes linspace / shift / stretch / invert / terminal entry by entry, and compares against the reference:

- **Example values**: at \(N=4\), all six configurations (static, exponential, linear, terminal, invert, terminal+invert) have timesteps and sigmas hard-coded to six decimals;
- **Equivalence**: static shift vs linear \(\mu=\)shift vs exponential \(\mu=\ln\)shift agree element-wise for \(N\in\{2,4,7\}\) and shift\(\in\{1,1.5,2,5\}\); shift=1 and exponential \(\mu=0\) are both identities;
- **Monotonicity and endpoints**: under the three shift types plus terminal, the sigmas are non-increasing (endpoints pinned to 1 and 0 without a terminal), and timesteps always equal `sigmas[:-1]*T`;
- **Terminal ordering**: the last entry equals `shift_terminal`, the whole row equals \(1-(1-z)/\text{scale}\), the last timestep is `terminal*T`, and the wrong-order variant is asserted to deviate by more than 1e-3 while the first entry moves towards the noisy end;
- **Inversion**: `sigmas[:-1] == 1 - original sigmas[:-1]`, the terminal is 1, timesteps are recomputed from the inverted sigmas, and the row is increasing;
- **Sigma range and \(N=1\)**: the six-decimal result for `sigma_min=0.1/sigma_max=0.9`, and a single timestep at \(N=1\);
- **Scale**: with `num_train_timesteps ∈ {1,10,1000,4096}` the timesteps scale exactly in proportion;
- **Randomised combinations**: 8 configurations at \(N\in\{1,2,3,5,8\}\) (including negative \(\mu\) and a narrow sigma range with terminal + invert) match the oracle element-wise;
- **Invalid inputs**: 18 groups (\(N\) as 0/float/bool, non-positive or NaN shift, unknown shift type, dynamic shifting without \(\mu\), \(\mu=\infty\), linear \(\mu=0\), terminal of 0/1/negative, \(N=1\) with terminal, illegal sigma bounds, \(T=0\)) must all raise `ValueError`.

Nine assertion groups in total.

## Complexity

\(O(N)\) time and space: two element-wise algebraic transforms and one append.
