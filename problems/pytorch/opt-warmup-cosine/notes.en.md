# Linear Warmup + Cosine Annealing Learning Rate Explained

## Source
"warmup + cosine" is the default learning-rate recipe for Transformer/ViT training: SGDR by Loshchilov & Hutter (arXiv:1608.03983) introduced cosine annealing; Goyal et al. (arXiv:1706.02677) introduced linear warmup. The combination is implemented by stitching `torch.optim.lr_scheduler.LinearLR` and `CosineAnnealingLR` together through `SequentialLR` — this problem asks you to **compute each step's \(lr\) yourself from the closed-form formula**, without assembling ready-made schedulers.

## Core idea
- **Linear warmup**: over the first few steps, raise \(lr\) linearly from \(base\_lr/\text{warmup}\) to \(base\_lr\). Its purpose is to let Adam-like optimizers accumulate their second-moment estimate first, avoiding a large early \(lr\) that amplifies noisy gradients.
- **Cosine annealing**: then smoothly lower from \(base\_lr\) to \(min\_lr\) following \(\tfrac12(1+\cos(\pi\cdot\text{progress}))\); `progress` goes from 0 to 1. Compared with step-wise decay, cosine is smoother and converges more finely near the end.

## Why it is easy to get wrong (pitfalls)
1. **The reference point of \(t\)**: right after construction it should already be at step \(0\) (\(lr=lr(0)=base\_lr/\text{warmup}\)), not \(lr(0)=base\_lr\). After the \(k\)-th `step()` call it enters step \(k\) (\(k=1,2,\dots\)).
2. **warmup uses \(t+1\), not \(t\)**: with \(t/\text{warmup}\), step \(0\) would have \(lr=0\), which does not match the reference. In this formula step \(0\) is \(base\_lr/\text{warmup}>0\).
3. **Join at the peak**: the end of warmup (\(t=\text{warmup}-1\)) and the start of cosine (\(t=\text{warmup}\)) both equal \(base\_lr\); the two 0.1 values in the example are normal, not a bug.
4. **The denominator of `progress` and clamping**: the denominator is `total_steps - warmup_steps` (the length of the cosine segment), not `total_steps`; and it must be `clamp`ed to \([0,1]\) to prevent \(lr\) from rebounding when \(t\) exceeds the schedule. In the degenerate case `denom<=0`, take \(lr=min\_lr\) directly.
5. **Not using `LRScheduler.__init__`'s bookkeeping**: the official `LRScheduler` automatically calls `step()` once during construction and maintains `_step_count`, whose semantics differ from this problem's "construction is step 0, only step advances". The reference solution manages its own `_t` counter, writes \(lr(0)\) directly at construction, and inside `step()` does `_t+=1` then writes back, avoiding conflicts with the parent class's bookkeeping.

## Grading details
The grader uses an independent pure-Python recomputation of the closed-form `lr(t)` (`_ref_lr` in `test_cases.py`, computed segment by segment with the standard `math` module, structurally different from the reference solution), driving the scheduler object and reading `param_groups[0]["lr"]` step by step, comparing the whole sequence with `atol=1e-6, rtol=1e-6`. Coverage: the example parameters, `warmup_steps=0` pure cosine, different hyperparameter combinations, and lower-bound clamping (the last step should be ≈`min_lr` and the cosine segment monotonically non-increasing).

## Complexity
\(O(1)\) per step; no extra tensor state, only a counter field.
