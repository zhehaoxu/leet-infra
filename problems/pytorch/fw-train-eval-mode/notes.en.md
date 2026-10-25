# Explained: Bookkeeping for train/eval Modes, Inference and Grad Clipping

## Origin

PyTorch's `torch.nn.Module.train/eval` docs: the mode flag only changes the behavior of mode-sensitive modules such as BatchNorm/Dropout — it never touches autograd; `nn.BatchNorm1d` documents that train-mode forwards use batch statistics and update \(running\leftarrow(1-m)running+m\,batch\) (with the unbiased batch variance), and that `num_batches_tracked` increments only in training. The `torch.inference_mode` docs state that inference tensors produced inside the mode do not support autograd and that re-enabling them in place via `requires_grad_(True)` outside the mode raises — in contrast to `no_grad`, whose outputs are ordinary tensors that can be re-enabled. The `clip_grad_norm_` docs: with `norm_type=inf` the total norm is the max absolute value over all gradients, and only if `total_norm > max_norm` every gradient is multiplied by `max_norm/(total_norm+1e-6)`.

## Key points and pitfalls

1. **eval() ≠ no gradients**: after `bn.eval()` the affine parameters still have `requires_grad=True`, so the eval-mode output carries a grad_fn. Doing inference with only `eval()` and no `no_grad/inference_mode` needlessly records the whole graph.
2. **BN's running ledger is unbiased + EMA**: batch variance divides by \(N-1\); both updates share momentum=0.1. Here every channel's two batches have \(|b-a|\) of 0.75/2.5, so the unbiased variances are \(0.28125/3.125\) for all three channels and running_var is identical across channels — a hand-computable anchor.
3. **Dropout uses inverted scaling**: in train, kept elements are multiplied by \(\frac{1}{1-p}\), so an all-ones input has value set exactly \(\{0,\frac{1}{0.75}\}\); eval returns the input untouched. The count \(8/20\) depends on the Bernoulli stream from seed 7 — asserted exactly only for the example, with invariant checks for other solutions.
4. **Two walls around inference tensors**: `.backward()` raises "element 0 of tensors does not require grad and does not have a grad_fn"; `.requires_grad_(True)` outside the mode raises "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed". `no_grad` outputs only hit the first wall (and can be re-enabled).
5. **Clipping scales uniformly, it does not truncate elementwise**: \(-4\) and \(-0.5\) are multiplied by the same \(coef=1/(4+10^{-6})\); even small gradients shrink (1.0→0.24999994). The +1e-6 guard makes the clipped max norm strictly below max_norm.
6. **Eval forwards freeze the state**: running stats and `num_batches_tracked` are untouched — the minimal checkable form of "eval polluted my training statistics" bugs.

## Cross-check basis

The solution uses the real framework; the `_oracle` is pure Python: per-channel \((1-m)rm+m\,\bar{x}\) and unbiased \(\sum(x-\bar{x})^2/(N-1)\) for the running ledger, the closed form \((x-rm)/\sqrt{rv+10^{-5}}\) for the eval output, and elementwise multiplication by \(coef\) for the clip ledger — all matched with `allclose(1e-5,1e-5)`. The dropout count cannot be reproduced in pure Python (torch's Philox stream), so the example asserts 8 exactly while other checks use the \(\{0,4/3\}\) value set and \(0<n<20\) invariants. Booleans, counters and total_norm are asserted exactly.

## Complexity

\(O(B\cdot C+P)\): B=2 batches × 3 channels plus P=6 gradient elements — constant size.
