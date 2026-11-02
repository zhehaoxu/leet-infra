# notes (English)

## Upstream

- Repo: `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_k_dpm_2_discrete.py` (715 lines, downloaded 2026-11-02 via the contents API with `Accept: application/vnd.github.raw`). The ancestral variant `scheduling_k_dpm_2_ancestral_discrete.py` (750 lines) was cross-checked to confirm the up/down split is out of scope here.

## Line-by-line mapping

| Upstream | This problem |
| --- | --- |
| `set_timesteps` L323-327 | After appending the trailing 0, `sigmas_interpol = sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()` — a log-space interpolation against `roll(1)` (the previous sigma) = the geometric mean of adjacent sigmas. This problem implements the isomorphic `√(σ[k−1]·σ[k])` (√(ab) = exp((log a + log b)/2), differing only in the last ulp; the oracle is self-consistent) |
| `set_timesteps` L329-332 | Expanded tables: `cat([t[:1], t[1:].repeat_interleave(2), t[-1:]])`, length 2N+2. Structure `sigE[2k]=σ_k`, `sigE[2k+1]=σ_{k+1}`, three trailing zeros |
| `set_timesteps` L343-345 | `interleaved_timesteps = stack((timesteps_interpol[1:-1], timesteps[1:])).flatten()` — timesteps length **2N−1**, one less than sigmas → **the last logical step has only the first call** (the same "sigmas one longer than timesteps" argument as problem 643) |
| `step` L601-615 sigma picks | First call: `sigma = sigE[i]`, `sigma_interpol = sigIE[i+1]`, `sigma_next = sigE[i+1]` (fetched but unused); second call: `sigma = sigE[i−1]`, `sigma_interpol = sigIE[i]`, `sigma_next = sigE[i]` |
| `step` L614-615 | `gamma = 0` hardcoded, `sigma_hat = sigma·(gamma+1)` identity; not parameterized here |
| `step` L618-631 x0 | epsilon / v_prediction branches kept verbatim; `sample` type raises NotImplementedError upstream → folded into unknown-type ValueError here |
| `step` L633-640 first call | `derivative = (sample − x0)/sigma_hat`, `dt = sigma_interpol − sigma_hat`, `self.sample = sample` (**stores only the sample**, not dt/derivative) |
| `step` L641-650 second call | Comment literally reads `# DPM-Solver-2`: `derivative = (sample − x0)/sigma_interpol`, `dt = sigma_next − sigma_hat` (**full step**), `sample = self.sample` (rollback) then cleared |
| `step` L655 | `prev_sample = sample + derivative * dt` shared by both branches |

## Upstream quirks (faithfully kept/documented)

1. **`I[0]` is never read**: on the torch path `log(0) = −inf` enters the lerp, giving `I[0] = exp(nan) = nan` and `I[N] = exp(−inf) = 0`; the numpy path is the reverse (`I[0]=0`, `I[N]=nan`). After expansion the nan lands at `sigIE[0]` (torch) — every step read index is ≥ 1, so it is **never consumed**. The ancestral variant outright adds `sigmas_interpol[-2:] = 0.0` (L337) to clean up — direct evidence the authors hit this. The geometric-mean construction here yields 0 naturally, matching the torch path at every read index.
2. **The last logical step lands on zero in one call**: `sigIE[2N−1] = I_N = 0`, so the final first-call has `dt = −σ_{N−1}`, and under epsilon `prev == x0` (within 1 ulp).
3. **The first call's `sigma_next = sigE[i+1]` is fetched and discarded** — the root of the m-b-next-idx equivalence (see below).

## The three mechanical differences vs Heun (problem 643), one mutation each

| | Heun (643) | KDPM2 (this problem) |
| --- | --- | --- |
| Midpoint | arithmetic half-step (sigma pair reused, no midpoint table) | log-σ geometric mean `√(σ_j·σ_{j+1})` |
| Derivative combination | `(d_first + d_second)/2` | midpoint derivative only; the first is discarded |
| Second-call dt | the stored half-step dt | full step `σ_{j+1} − σ_j` |
| Second-call derivative denominator | σ_next | the midpoint m |
| state | three keys (derivative+dt+sample) | one key (sample) |

## Deliberate deviations (documented)

- **No diffusers dependency**: the base sigma table (unexpanded) is an explicit input; the expanded tables are rebuilt inside solve on every call (pure function, no `self` state; state passed explicitly as a dict).
- **Extra structure validation**: last entry exactly 0, first N entries > 0, first-call even / second-call odd, second-call ≤ 2N−3, first-call ≤ 2N−2, midpoint > 0. Upstream validates none of these (pipeline convention); they encode the real "pipeline loops over 2N−1 timesteps" pattern.
- Geometric mean computed directly with `sqrt` rather than log-lerp: differs only in the last ulp and sidesteps the torch/numpy divergence at −inf (quirk 1).

## Test design

- Independent numpy oracle `_oracle` rewritten from set_timesteps L323-345 + step L598-663 (including table construction); 13 test cases ACCEPTED in both modes (run 2/2 + submit 13/13).
- Analytic anchor cases: the `[4,1,0]` full trajectory 0.8 → 0.4 → 0.5 (hand-computed, not oracle cross-check); non-square geometric midpoint `√(3·2)`; degenerate `[2,2,0]` (all dt zero); hand-computed v_prediction for both calls.
- 6 random cross-checks (N ∈ 2..4, random σ/mo, fixed seed) comparing prev/x0/dt per call; tolerance 1e-6 — **solve round-trips through float32** (pipeline semantics) while the oracle is float64, so 1e-9 false-fails.
- Pitfall: the final call (a first call) returns `state = {"sample": ...}`, **not None** — upstream never clears `self.sample` at the end; the test initially asserting a None state at the end was wrong.
- Pitfall: a test once passed the oracle's numpy state back into solve (it must be `out["state"]`, a torch tensor) — solve rejected it with the "state.sample must be a torch tensor" message.

## Mutation testing

35 mutations: **30 CAUGHT + 5 equivalent** (equivalences proven and recorded below).

Equivalent 5:

1. `m-geom-shift`: `(v[:-1]*v[1:])` → `(v[1:]*cat([v[:1], v[1:-1]]))` — element-wise the same products (k=0 term `s1·s0`, k≥1 terms `s_{k+1}·s_k`); 200 random sigmas give 0 differences under `torch.equal`;
2. `m-head-not-zero`: midpoint table head `[0]` → `[I1]` — `I[0]` only lands at `sigIE[0]`, every step read index is ≥ 1, never consumed (**exactly mirroring upstream quirk 1**: the official torch path has nan there anyway);
3. `m-no-tail`: drop the expanded-table tail pad (length 2N+1) — `sigE[2N]`/`sigE[2N+1]` have no readers (max read index 2N−3 for B, 2N−2 for A);
4. `m-ie-uses-sig-e`: `sigIE` tail uses `values[-1:]` (=0) instead of `interpol[-1:]` (=I_N=√(σ_{N−1}·0)=0) — identical by construction;
5. `m-b-next-idx`: second call `sigma_next = sigE[i]` → `sigE[i+1]` — under the paired structure `sigE[2j+1] = σ_{j+1} = sigE[2j+2]`, identical for every reachable odd i (1..2N−3); 200 random checks, 0 differences.

Key catches (strengthened after first-round SURVIVED):

- `m-parity-b` (drop the second-call parity check): the original even-index second-call cases all fell outside the range check and were **vacuously caught** — added "N=3, i=2, with state" (the mutant silently computes a wrong value);
- `m-a-range-loose` (upper bound 2N−2 → 2N): the original i=5 case still raised under N=2 — added "i exactly 2N" (the mutant divides by σ=0, returns nan without raising);
- `m-b-range-loose` (upper bound 2N−3 → 2N−1): the mutant's last-step second call still raises via the "midpoint must be positive" check, vacuously caught — switched to **asserting the range-error message** (the one containing the "step_index must fall within" wording) rather than any ValueError.
