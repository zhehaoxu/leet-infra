# Solution Notes · Entropy-Bounded Unmasking

## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_entropy_bound.py`: `EntropyBoundScheduler.step` (lines 118–182) is the prototype. Its comments state it serves uniform-corruption discrete diffusion — no MASK token, unreceived positions are simply replaced with uniformly random tokens.
- The paper is given in the class docstring: *Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking* (arXiv:2505.24857). The paper bounds the joint mutual information among accepted tokens; after sorting, each position only needs to drop its own entropy term, which is how the engineering form `cum - H_self <= bound` arises.
- In the source, candidates are drawn live with `torch.multinomial` (and the step also returns `sampled_probs` and `pred_logits`). This problem injects the candidate and random tokens as **inputs**, removing the RNG from the contract and keeping only the deterministic part.

## Core Points and Pitfalls

**The prefix sum must exclude the position itself.** The source writes `cumulative_entropy - sorted_token_entropy <= entropy_bound`: `cumsum` includes the current term, so subtracting it gives "the sum over all strictly lower-entropy positions". Writing `cumsum <= bound` accepts nothing at \(B=0\) (the first prefix already equals its own entropy) and is one position short in general — the \(B=0\) test targets exactly this.

**The temperature scales the logits, and the entropy uses the scaled distribution.** The source first does `model_output = model_output / temperature`, then uses that same tensor for both sampling and entropy. Entropy is not temperature-invariant (except for the uniform distribution), so computing it from raw logits makes the annealing a no-op. The test pins the temperature to 1 with `t_max=t_min=1.0` and compares against a hand-written \(-\sum p\log p\), isolating "is the scaling applied to the entropy".

**Annealing direction: a smaller `timestep` is hotter.** `fraction = (N - timestep) / N`, so `timestep=0` gives fraction 1 and temperature \(t_{\max}\); the last step only approaches \(t_{\min}\), landing at \(t_{\min}+(t_{\max}-t_{\min})/N\). Treating `timestep` as "remaining steps" or writing `timestep / N` flips the whole curve.

**Pin down the tie semantics.** `torch.sort` is not stable by default, so the order of equal entropies is unspecified. This contract fixes ties as "smaller index first", the solution uses `stable=True`, and the oracle sorts by `(entropy, index)`. Otherwise identically distributed positions swap acceptance whenever \(B\) sits exactly on a boundary.

**Acceptance order and output order are two different indexings.** `sorted_indices` is only used to decide in sorted order and to scatter back; the last dimension of `prev_sample`, `accepted` and `token_entropy` must return to the **original** positions. Applying `where` directly to `sorted_accepted` yields a token block permuted by entropy.

**Numerically stable entropy.** Computing `p * log(p)` naively turns `0 * -inf` into `nan` for extreme logits. Using `log_norm = logsumexp(z)`, `log_p = z - log_norm`, `H = -Σ exp(log_p) * log_p` makes the vanishing terms disappear; `Categorical(logits).entropy()` follows the same path internally.

**Type conventions.** The mask is bool, tokens are int64 (the source uses LongTensor), entropy is float32; compute the entropy internally in float64 so float32 rounding does not accumulate across the per-position cumsum.

## Division of Labor

- `diff-ddpm-step` / `diff-ddim`: one-step updates in continuous Gaussian space with float tensor state; here the state is discrete token ids, there is no variance term, and entropy decides how many positions can be fixed this step.
- `diff-repaint-jump-schedule`: also answers "where to go next / which positions to redo", but that is a deterministic calendar over timesteps in image space; this problem decides the accepted set dynamically from the current distribution's entropy.
- `dt-unigram-sample`: also "sample discrete tokens from a distribution", but for tokenizer training/encoding; here it is a batched commit at decoding time.
- `diff-v-prediction-convert`: conversion between continuous prediction targets, with no overlap with the discrete acceptance rule here.
