# Bookkeeping for train/eval Modes, Inference and Grad Clipping

## Problem

`model.eval()` only flips the behavior of mode-sensitive operators like BatchNorm/Dropout — it does not switch off autograd; `torch.inference_mode()` additionally disables gradient recording entirely, and the inference tensors it produces cannot even be re-enabled with `requires_grad_(True)` afterwards. `clip_grad_norm_(..., norm_type=inf)` uses the max absolute value as the total norm and multiplies every gradient by one uniform coefficient. This problem turns these mode behaviors into bitwise-checkable ledgers across five stages.

## Rules

Implement `solve()`, run the five stages from the starter docstring, and return the behavior ledger dict:

1. **Stage A**: default `nn.BatchNorm1d(3)` (momentum=0.1, eps=1e-5) with two train forwards; record running_mean/running_var (unbiased batch variance, \(running\leftarrow0.9\,running+0.1\,batch\)) and `num_batches_tracked`.
2. **Stage B**: after `bn.eval()`, forward `xe`; verify the state_dict tensors are untouched bitwise, and hand-check the normalized output computed from the running stats.
3. **Stage C**: `torch.manual_seed(7)`, `nn.Dropout(0.25)` on an all-ones input: in train the value set is exactly \(\{0,\ \frac{1}{0.75}\}\) (inverted scaling), in eval it is the identity.
4. **Stage D**: compare three "no-grad" routes: an eval-mode forward output still `requires_grad`; an `inference_mode` output cannot backward and cannot be re-enabled in place; a `no_grad` output can be re-enabled.
5. **Stage E**: manually fill `.grad` on two Parameters, then `clip_grad_norm_([w1, w2], max_norm=1.0, norm_type=inf)`: total norm = max absolute value \(=4.0\), coefficient \(coef=1/(4+10^{-6})\) applied to every element.

## Examples

```
Input: none
Output:
{
 "running_mean_after2": [0.058750003576278687, 0.13124999403953552, 0.2537499964237213],
 "running_var_after2": [1.1478124856948853, 1.1478124856948853, 1.1478124856948853],
 "num_batches_train": 2,
 "eval_state_unchanged": true,
 "eval_out_requires_grad": true,
 "eval_out_flat": [0.41185787320137024, -1.055896282196045, 1.6299304962158203,
                   0.8785523176193237, 0.11083994060754776, -0.7035419344902039],
 "dropout_train_unique": [0.0, 1.3333333730697632],
 "n_dropout_zeros": 8,
 "dropout_eval_identity": true,
 "inference_requires_grad": false,
 "inference_backward_raises": true,
 "inference_enable_raises": true,
 "no_grad_enable_ok": true,
 "total_norm_inf": 4.0,
 "grads_after_flat": [0.7499998211860657, -0.9999997615814209, 0.0,
                      0.49999988079071045, 0.24999994039535522, -0.12499997019767761],
 "clip_scaled_uniformly": true
}
```

Each channel's unbiased batch variance is exactly \((b-a)^2/2\in\{0.28125\,(x_1),\ 3.125\,(x_2)\}\) for all three channels, hence the identical running_var entries.

## Constraints

- Running stats, eval outputs and clipped gradients use `atol=1e-5, rtol=1e-5`; booleans, `num_batches_train`, `n_dropout_zeros` and `total_norm_inf` must match exactly.
- Stage C may only rely on the determinism of `manual_seed(7)`; do not substitute a fixed mask.
- Stage D's three raises/ok flags must come from actually catching (or not catching) `RuntimeError`, not hard-coded constants.
