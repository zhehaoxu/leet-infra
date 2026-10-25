# In Depth: WSD Warmup-Stable-Polynomial Decay

## Source
- The three-phase WSD form and its use in LLM pretraining: Hu et al., *MiniCPM: Unveiling the Potential of Large Language Models with Scalable Training Strategies*, arXiv:2404.06395 (the paper states "Warmup-Stable-Decay (WSD) learning rate scheduler, conducive to continuous training and domain adaptation").
- Systematic analysis of WSD (the constant branch plus a terminal decay, versus cosine): Wen et al., *Understanding Warmup-Stable-Decay Learning Rates*, arXiv:2410.05192.
- Reference implementations: HuggingFace `transformers/optimization.py::_get_wsd_scheduler_lambda` / `get_wsd_schedule` (`warmup_type/decay_type ∈ {linear, cosine, 1-sqrt}`, where `linear` is exactly \((1-t)^1\)); kozistr/pytorch_optimizer `pytorch_optimizer/lr_scheduler/wsd.py` (cooldowns `linear`/`1-sqrt`/`1-square`, the same family).
- Note: the arXiv:2110.07621 recorded in todos.md is **not** the WSD paper (it is a hydrodynamics article by Heller et al.); this problem is built on the three verifiable sources above.

## Core points and pitfalls
1. **The three lengths**: `warmup_steps`, `stable_steps`, `decay_steps` are *lengths* and the phase boundaries are half-open: \(t\in[0,W)\) warmup, \([W,W+S)\) stable, \([W+S,W+S+D)\) decay, floor afterwards. The stable phase is the soul of WSD: it can be extended indefinitely and the decay only happens at the very end, which is what makes "one constant branch plus many decay branches" from a single checkpoint possible.
2. **The decay is \((1-t_{\text{decay}})^p\), not \(1-t_{\text{decay}}^p\)**. These are different curve families: they coincide at \(p=1\), but at \(p=2\) the former is \((1-\tau)^2\) (zero slope, soft landing) while the latter is \(1-\tau^2\) (steep at the end). This is the easiest thing to transpose.
3. **The floor is a ratio**: `min_lr_ratio` is not `min_lr` but a multiplier. Warmup starts at `base_lr·m` (at \(t=0\) it is *not* 0) and decay ends at `base_lr·m`, via \(f(1-m)+m\); the stable phase stays a pure 1.
4. **Step convention**: "construction is already step 0", `step()` advances — the same convention as `opt-warmup-cosine`. Do not shift the sequence by one by reading before/after the increment inconsistently.
5. **Division by zero**: with `decay_steps=0` you must not compute \(\tau\) and go straight to the floor branch; with `warmup_steps=0` the very first step is already in the stable phase.

## Cross-check convention
- Primary: an **independent pure-Python recomputation** of \(\text{base\_lr}\cdot f(t)\) (the `_oracle` in test_cases uses the different structure "raw multiplier first, single clamp afterwards"), point-wise absolute error < 1e-9.
- Hard reference: \(p=1\) is equivalent to `transformers.get_wsd_schedule(warmup_type="linear", decay_type="linear")`, used as an external oracle step by step (1e-12). HF's `LambdaLR` also writes the \(t=0\) learning rate at construction, so the sequence indices align naturally.
- Shape assertions: strictly increasing during warmup, exactly equal during stable, monotone non-increasing during decay, and constantly `base_lr·min_lr_ratio` past the total.

## Complexity
\(O(1)\) per step (one closed-form evaluation per param group), no extra state; the whole \(W+S+D\) accounting lives in the index, not in tensors.
