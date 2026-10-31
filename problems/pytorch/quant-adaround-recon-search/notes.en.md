# Solutions · AdaRound Reconstruction-Error Candidate Search

## Sources

- AdaRound (Nagel, Faronato, Diniz, Corradi, Galesso & Bassett, ICML 2020, **arXiv:2004.10568**, "Up or Down? Adaptive Rounding for Post-Training Quantization"; corrected on 2026-10-30 - it was recorded as 2004.10501, which is actually Graubohm et al.'s driverless-vehicle concept-phase paper): every weight gets a searchable softening variable h, the reconstruction error on calibration activations decides which grid point to move toward, and an annealed exponent pushes solutions to the ends.
- The line-level formulas here are transcribed from `training_utils.py::AdaRound` in `Saad-Fahmy/LLM-QAT` (which follows the upstream `ideal/AdaRound` implementations): `floor(w / scale)` for the grid point, the `K0 = -0.1 / K1 = 1.1` linear interpolation, the `floor(h * resolution) / resolution` ladder decode, `alpha = sigmoid((logit(bar_h) + log(resolution)) / 2)`, the greedy search over the candidate offsets `[0, +1/2**q, -1/2**q, +1, -1]`, and the `|h - 0.5| <= 0.5 / resolution` decision band.
- **Re-check record for the transcribed repository**: `training_utils.py::AdaRound` in `Saad-Fahmy/LLM-QAT` was checked twice inside this session - one check **fetched raw HTTP 200 and matched all 121 lines of that class against the formulas above** (the candidate table, the decision band, `logit(bar_h)`, the divide-by-2 temperature, \(p=1+(i+1)/T\), `lam=5.0` and the `|alpha-0.5|` penalty are all present verbatim), the other check got 404 on the same path (LLM-QAT now lives under `facebookresearch/LLM-QAT`, so raw flakiness or the rename intervenes). The formulas above therefore do have a source, and the problem statement remains the sole specification for judging. **Independently corroborated backbone**: `yhhhli/BRECQ/quant/adaptive_rounding.py` and `quic/aimet`'s `adaround_*` confirm \(\gamma=-0.1,\ \zeta=1.1\), `x_floor = floor(x/delta)`, `rest = x/delta - x_floor`, the logit initialisation \(\alpha=-\log\big(\frac{\zeta-\gamma}{\text{rest}-\gamma}-1\big)\), the soft target `clamp(sigmoid(alpha)*(zeta-gamma)+gamma, 0, 1)` and the hard decode `x_int = x_floor + (alpha >= 0)`; on the AIMET side the reconstruction term \(\lVert \hat Y-Y\rVert_F^2\), the regulariser \(1-|2\alpha-1|^{\beta}\), the cosine beta annealing `end + 0.5*(start-end)*(1+cos(rel_iter*pi))` and the rule that the regulariser is off before `warm_start`. **Pinned by this exercise only** (no reachable implementation carries them): the `resolution` ladder combined with the `+log(resolution)` shift and divide-by-2 temperature, and the **per-element five-candidate greedy search** (the paper's `loss_grid`; its body was unreachable this round).

## Source transcriptions and deliberate simplifications

- **The stochastic sampling branch is dropped entirely.** Real AdaRound draws each round's candidate weights from a Bernoulli (temperature annealed) sampler; this problem replaces it with a **deterministic greedy candidate search**, so the same input is bit-for-bit reproducible and judging can look only at the return value.
- **The layer-wise sequential block reconstruction schedule is dropped.** Upstream searches one layer, freezes it, then forwards into the next layer and propagates the error; here there is a **single** `W`, `X` is an explicit small matrix and no model forward pass or trained network is involved.
- **`zeta` and `log2h` in `betas_range = (log2h / zeta) / log(lam)` are excluded.** In the transcribed source `zeta` is never defined and `log2h` (log2 of h) is an **elementwise** quantity, not a **per-step** scalar, so it cannot be the per-round regularizer weight as written. Rather than invent a reading, both symbols are removed and only the unambiguous pieces survive: the \(1/\log(\text{lam})\) factor and the annealing exponent \(p_i = 1 + (i+1)/T\), giving \(\beta_i = p_i/\log(\text{lam})\). The whole `betas` list is exposed as an output key.
- **The residual must be normalized before interpolation.** The transcription reads `h = clamp((r - K0)/(K1 - K0))`, but \(r\in[0,\text{scale})\) does not share units with \(K_0,K_1\in[-0.1,1.1]\); copying it literally clamps every weight with \(r\) not tiny to \(h=1\) and the search degenerates. Upstream really interpolates the residual measured **in step units**, so this problem pins \(z = r/\text{scale} = W/\text{scale} - q\in[0,1)\).
- **The \(q=0\Rightarrow h=1\) override is pinned as a formula, not as a comment.** The transcription's side note calls it "forced rounding down", yet \(\alpha\) is the **round-up** probability and \(h=1\) saturates \(\alpha\) toward 1 (next bullet), so the two statements are not self-consistent on this code path. The statement fixes the override and its numeric effect; example 2 keeps it visible in the final \(h=1\).
- **The \(+\log(\text{resolution})\) shift and the divide-by-2 temperature are copied as-is**, so \(\alpha\) is strongly biased toward 1: at \(\bar h = 0.5\), \(\alpha = \text{sigmoid}(\tfrac12\log 1024) = 32/33 \approx 0.9697\), and at \(\bar h=\varepsilon\), \(\alpha\approx 0.031\). Candidate ordering is mostly driven by this "compressed toward 1" bias; it is not a bug.
- **\(\varepsilon=10^{-6}\) has exactly two fixed jobs**: clamping \(\bar h\) into \([\varepsilon, 1-\varepsilon]\) before the logit (otherwise it diverges) and the tolerance of the final mask band \(0.5/\text{resolution}+\varepsilon\). One constant, no extra eps.
- **`searched_mse` is the reconstruction term only**, without the regularizer; the regularizer only participates in the search and in `soft_loss`.

## Core ideas and common mistakes

**Interpolate z, not r.** \(h=((z+0.1)/1.2)\) lands in \([0,1]\); using \(r\) directly clamps almost everything to \(h=1\) and the search becomes "nobody moves".

**Ladder decode before clamping.** \(\bar h=\lfloor h\cdot\text{resolution}\rfloor/\text{resolution}\) first, `clamp(eps, 1 - eps)` after; clamping first turns \(h=1\) into \(\bar h = 1-2^{-10}\) and loses a whole ladder step of \(\alpha\).

**The freeze band and the final mask share \(0.5/\text{resolution}\) but mean different things.** The former decides whether the element is searched again next round; the latter decides whether the final decode uses \(\bar h\ge 0.5\) or \(\operatorname{round}(\alpha)\). Only elements still inside the band when `steps` runs out take the mask branch.

**The objective is over the whole `W`.** While trying a candidate, the reconstruction term sums over **all** output rows and the regularizer averages over **all** elements; writing "current element only" changes which candidate wins as soon as there is more than one column (the mean's denominator becomes \(On\) instead of 1).

**Ties keep the earlier candidate, update only on a strict improvement.** After clamping to \([0,1]\), `0`, \(+1/2^{q}\) and \(+1\) often give the same trial \(h\) with bit-identical objectives, and `0` must win: the `chosen_offset = 0` of example 2 is exactly this rule.

**`chosen_offset` is the last commit.** A band element is searched in several rounds and each round overwrites it; row-major visiting plus per-round commits, with the order reversed, yields a different \(h\).

**`q` and `r` are fixed by the input, not by the search.** They are decided before any round; `h`, `alpha` and `chosen_offset` are post-search state - do not return the initialization.

**`betas` is a \((T,)\) scalar list**, \(\beta_i=p_i/\log(\text{lam})\). With \(\text{lam}<1\), \(\log(\text{lam})<0\) and the regularizer weight flips sign (it then rewards the extremes); that is a legal input, do not take `abs()`.

```
L_i = sum_o ||X * ((q + alpha)_o - W_o / scale) * scale||^2 / n + beta_i * mean(|alpha - 0.5| ** p_i)
candidates = [0, +1/resolution, -1/resolution, +1, -1]   # ties keep the earlier candidate
freeze     = abs(h - 0.5) > 0.5 / resolution             # after the commit
mask       = abs(h - 0.5) <= 0.5 / resolution + 1e-6     # before the final decode
```

## Cross-check setup

- Reference: `variants/pytorch/solution.py`, torch tensors, float64 throughout; each round visits unfrozen elements row-major and clones `h` once per candidate, then re-decodes and re-scores the whole tensor.
- Independent oracle: `test_cases.py::_oracle`, pure Python scalar loops (`.tolist()` out, elementwise `math.floor/log/exp`, reconstruction accumulated over \((o,t,j)\)), sharing no control flow with the reference. The two agree on all nine keys over 120 random instances (\(O,n,T_\text{cal}\le 4\), `num_iters` 1..6, `steps` 1..4, `lam` from 0.3/1.05/5.0/1000, `scale` from 0.25/1.0/3.0, with one activation column randomly amplified): int64 exactly equal, floats within 3e-5 in float32.
- Targeted cases cover: both public examples, the candidate tie resolved to `0`, the \(q=0\) override, an exactly-on-grid element (\(r=0\)), a final-mask hit (\(h\) inside the band with \(\bar h = 0.5\) rounding up), multi-round re-searching of band elements, the hand-checkable \(X=I\) reduction, a dtype sweep (float32 versus float64 agreement), non-contiguous inputs, inputs not mutated in place, shape/dtype invariants and three `ValueError` probes.

## Complexity

- Five whole-tensor scorings per unfrozen element per round: \(O(\text{steps}\cdot On\cdot 5\cdot(T_\text{cal}\cdot n+n))\), i.e. \(O(\text{steps}\cdot T_\text{cal}\cdot O\cdot n^{2})\).
- The problem sizes are deliberately tiny (\(n\le 4\), \(T_\text{cal}\le 4\)); brute force passes, and vectorization is only a constant-factor concern and never changes the answer.

## Related problems

- `quant-awq-clip-scale-search`: also "small given matrices plus an explicit search", but the search space is the per-channel scale-and-clip grid; here the search space is the round-up/round-down direction of each weight.
- `quant-smoothquant-alpha-search` / `awq-activation-scale`: single-parameter annealed search that migrates magnitude into the activations; AdaRound migrates the rounding direction.
- `group-wise-quantize` / `weight-only-int8-matmul`: the numerics of quantization itself (scale, zero point, code stream); here `scale` is an explicit input and no group calibration happens.
