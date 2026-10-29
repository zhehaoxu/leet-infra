# Notes: Three Engineering Conventions in Cross-Entropy Backward

## Origin

The backward of `F.cross_entropy` is the most-asked loss question in LLM pretraining and SFT interviews: it is simultaneously the collapsed form of "softmax backward + one-hot", and the place where label smoothing, `ignore_index` (padding must not enter the loss) and `reduction` meet. The bonus point is usually the denominator — many candidates divide the mean by N instead of the number of valid rows.

## Deriving p − y from Softmax Backward

With \(L=-\sum_k y_k\log p_k\) and softmax's backward \(dx_i=y_i(grad_i-\sum_k y_k grad_k)\), plugging \(grad_k=-y_k/p_k\) gives

$$dx_i = p_i\Big(-\frac{y_i}{p_i}-\sum_k y_k\Big(-\frac{y_k}{p_k}\Big)\Big) = -y_i + p_i\sum_k\frac{y_k^2}{p_k}$$

For a one-hot \(y\), \(\sum_k y_k^2/p_k=\sum_k y_k/p_k=1/p_{argmax}\) and the term cancels, leaving \(p_i-y_i\). In other words **\(p-y\) only holds for one-hot**; under label smoothing (\(y\) no longer 0/1) you must return to \(p_k-(1-\varepsilon)onehot-\varepsilon/C\) rather than writing \(p-\tilde y\) and dividing by C — those are two different \(\varepsilon/C\) spreading conventions.

## Core Ideas and Pitfalls

- **Denominator with ignored rows**: PyTorch's mean is `sum / max(valid_count, 1)`, so an all-ignored batch yields 0 instead of an error.
- **\(\varepsilon/C\) is spread over all C columns** including the true class, whose coefficient is \(-(1-\varepsilon)-\varepsilon/C\).
- **Numerical stability**: subtract the row max before exponentiating; ignored rows still need a well-defined softmax (their gradient is then zeroed) so that masking logits early cannot create `-inf - -inf`.
- **int64 target**: writing the one-hot with scatter beats per-row `arange` indexing, and using `ignore_index=-100` directly as an index goes out of bounds, so build the validity mask first.

## How It Differs from Neighboring Problems

`ag-softmax-backward` already receives the normalized output and needs one reduction; this problem starts from logits and additionally carries target plus three reduction conventions. `to-label-smoothing-ce` only computes the smoothed **loss value** (forward); here we need the **gradient**, and smoothing is coupled with ignoring: \(\varepsilon/C\) is added only on valid rows.

## Cross-Check Basis and Complexity

Judging uses `torch.autograd.grad(F.cross_entropy(logits, target, label_smoothing=..., ignore_index=..., reduction=...), logits)` as the oracle with atol=1e-9 in float64; the example uses all-zero logits (uniform distribution) so every entry is verifiable by hand. Complexity is \(O(NC)\) time and \(O(NC)\) extra space (a few constant factor with a reused softmax buffer).
