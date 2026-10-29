# Deep Dive: D3PM Uniform-Kernel Posterior Routes

## Where it comes from

The uniform corruption kernel of *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM, arXiv:2107.03006). The implementation follows HuggingFace diffusers' `src/diffusers/schedulers/scheduling_discrete_ddim.py`: inside `step`, `_sample_from_logits` draws greedy or temperature-scaled tokens, then `clean_mass / stay_mass / noise_mass` are stacked into a triple, divided by their own sum, and used to `multinomial`-pick a route; `_to_loo_logits` implements the leave-one-out denoiser correction. This problem keeps only the deterministic part at `temperature=0` (greedy); the random sampling is not judged.

## Key ideas and pitfalls

1. **The normalization is not decorative**: on positions with `same=False` the three masses sum to \((1-ab)/K\), so using them as probabilities directly under-counts. Upstream divides by the sum — reproduce that.
2. **Numerators and denominators of `a` and `b`**: `a = α_t/α_s` is the "survival probability from s to t" and `b = α_s`. Writing `a` as `α_s/α_t` breaks the deterministic commit on the final step (\(\alpha_s=1\), \(\alpha_t=1-1/N\)).
3. **The last step must commit deterministically**: \(\alpha_s=1\Rightarrow b=1\Rightarrow\) both stay and noise carry a \((1-b)\) factor and vanish, leaving clean alone normalized to a one-hot `[1,0,0]`. This is the "infinite terminal SNR" property that discrete and continuous DDIM share.
4. **`same` is a per-position indicator**: only positions that hit \(x_0\) receive the extra \(a\,b\) in clean. Using a global scalar or a whole-batch statistic is wrong.
5. **The LOO shift depends on time**: \(\delta=\operatorname{log1p}(K\alpha_t/(1-\alpha_t))\) diverges as \(\alpha_t\to1\); the source uses `log1p` for small-value accuracy and skips entirely when \(\alpha_t\le 0\) or \(\ge 1\). This problem's `step_index` range keeps \(\alpha_t<1\), but `step_index=0` gives \(\alpha_t=0\) and must skip.
6. **LOO damps exactly one logit — the observed token's**: it corrects the bias where the model over-reinforces a token it just wrote; other positions stay untouched. The additive direction matters too (the source uses `scatter_add` with a negative value, not an overwrite).
7. **Argmax tie semantics**: `torch.argmax` returns the first maximal index; the hand-written `_argmax` in the oracle must compare with strict `>` to agree.

## How the judge cross-checks

The reference computes in float64 and returns `route_probs` as float32 and `pred_tokens` as int64. The test `_oracle` uses no tensor ops at all: alpha accounting, the LOO shift, argmax, the three masses and the normalization are all recomputed per position with python scalars. Seven assertion groups: the three example steps plus the LOO before/after pair; one-hot commits on the final step (several N); the \([\alpha_s,0,1-\alpha_s]\) structure of the first step; the unnormalized sum equal to \((1-ab)/K\) where `same=False`; the LOO shift magnitude and the no-shift-at-`step_index=0` rule; nine (K, L, N, step, loo) combinations compared bit for bit; and invalid inputs. `pred_tokens` is compared exactly with `torch.equal`, `route_probs` with `atol=1e-6`.

## Complexity

An \(O(K)\) argmax per position plus \(O(1)\) mass arithmetic, i.e. \(O(BLK)\) in total, with no state and no extra buffers.
