# Deep Dive: The Row-Normalized Squared-ReLU Attention Forward

## Where it comes from

- Shen et al., 2023, arXiv **2309.08586**. The paper rewrites softmax attention as a family of point-wise attentions \(\phi=L^{-\alpha}h\) with \(h\in\{\mathrm{relu},\mathrm{relu}^2,\dots\}\) and \(\alpha\in[0,1]\), and reports that **squared ReLU** comes closest to softmax on vision Transformers. This problem takes the \(h=\mathrm{relu}^2\) branch.
- QK-Norm originates in Liu et al., *Understanding the Difficulty of Training Transformers*, 2020, arXiv **2010.04245**: normalize q and k along the head dimension before the dot product.
- Deliberate difference from the paper: the paper scales by the length constant \(L^{-\alpha}\), whereas this problem uses **row-sum normalization** \(\text{weights}=a/\max(\sum_j a_j,\texttt{eps})\). That makes the weights data-dependent and the row sums controllable, and removes the question of picking \(\alpha\). The normalization differs, but the "activate first, then normalize" skeleton is the same.
- Division of labour in the library: `logsumexp`/`softmax` are pure reduction problems, `nu-online-softmax-rescale` is about online softmax numerics, `attention-softcap` changes the logit cap function, and `layernorm-attention-scale` is about LayerNorm's coupling with \(1/\sqrt{h_d}\). Only this problem **removes softmax entirely**, replacing it with a point-wise activation plus row normalization, with QK-Norm on top.

## Core points and pitfalls

1. **ReLU first, then square — the order matters**. What is required is \(\mathrm{relu}(\text{logit})^2\); writing \(\mathrm{relu}(\text{logit}^2)\) collapses to \(\text{logit}^2\), so negative logits gain positive contribution. Both `check_activation_zeroing` and `check_not_shift_invariant` fail on it.
2. **Row normalization is not the paper's \(L^{-\alpha}\)**. The paper's factor depends only on sequence length and is a global constant; the denominator here is a data-dependent row sum. Do not implement both (e.g. row normalization plus an extra \(1/L\)).
3. **Degree-2 homogeneity kills the temperature exactly**. relu² is homogeneous of degree 2: scaling all logits by \(c\) multiplies numerator and denominator by \(c^2\), leaving the weights bitwise unchanged. So the \(\sqrt{h_d}\) temperature has no effect whatsoever, and multiplying q or k by any positive constant does not change the weights. `check_scale_invariance` verifies this with factors 7, 0.05 and 1e3 — it is a **property** of this attention family, not a degree of freedom in the implementation.
4. **But `eps` breaks point 3**. The denominator uses \(\max(\text{denom},\texttt{eps})\); when an entire row's logits are tiny (say q and k around \(10^{-4}\)) the true denominator falls below `eps`, the floor takes over, and the row sum shrinks below \(<1\). `check_zero_and_tiny_denominator` pins down this behaviour with a \(10^{-4}\) example.
5. **Zero logits get exactly zero weight**. Softmax would hand a zero logit \(1/n\); relu² gives it 0, which is the source of the activation's sparsity and tileability. A row with no positive logit gets all-zero weights and an all-zero output, never 0/0 and never a NaN.
6. **Shift invariance is lost**. Adding a constant to all logits leaves softmax unchanged but changes relu² drastically; `check_not_shift_invariant` builds a case where the shift turns a negative logit positive, first asserting that the change really is a uniform shift and that softmax is unchanged, then that this problem's weights must change.
7. **What qk_norm actually changes**. It applies a **gain-free** RMSNorm along the head dimension, \(x/\sqrt{\mathrm{mean}(x^2)+\texttt{eps}}\), keeping only the direction and discarding the norm: a key with 100× the norm no longer dominates, keys with the same direction get identical scores, and after normalization \(|\text{logit}|\le\sqrt{h_d}\). Note that `eps` also acts inside the RMSNorm, so "per-key rescaling invariance" is only approximate — the check uses a 1e-4 tolerance rather than bitwise equality.
8. **When the mask applies**. True = keep; it acts **after** activation and before the sum, zeroing masked positions so their weights are always 0. Fully masking a row is equivalent to a zero denominator, so that row is all zeros. The mask only needs to broadcast to the score matrix, not to match q/k's leading dims.
9. **Leading dims follow broadcasting rules**, e.g. q `(2, 3, L, h_d)` with k, v `(2, 1, S, ·)`; the output's leading dims are the broadcast result. The head dims of q and k must match, and k and v must agree on the sequence and leading dims.
10. **Precision convention**: compute internally in float64 and return float32; both float32 and float64 inputs must work.

## How it is judged

- `_oracle` is an independent numpy rewrite: `maximum(logits, 0) ** 2`, a `where` mask after `np.broadcast_to`, `maximum(denom, eps)`, and a final matmul — no code shared with the vectorized torch solution.
- Eleven `check_*` functions: the example, activation zeroing and row-sum properties, loss of shift invariance, squared growth (1:2:3 → 1:4:9), scale invariance, QK-Norm, zero and tiny denominators, mask and fully-masked rows, broadcast batches, randomized oracle configs (six, including mask, qk_norm and \(10^{-4}\) scales, plus a float32 input), and 21 invalid inputs.
- Mutation testing (square-then-relu, ignoring the mask, dropping the denominator `eps` floor, skipping the qk_norm RMSNorm, replacing row normalization with softmax, and others) is caught by at least one check. The only survivor is "drop the \(1/\sqrt{h_d}\) temperature", which degree-2 homogeneity plus row normalization makes an exactly equivalent transformation — a design property rather than a test gap.
- Invalid inputs only catch `ValueError`: non-tensors, integer/bool tensors, `ndim < 2`, NaN/±inf, an empty head dim, mismatched head dims, mismatched k/v sequence dims, non-broadcastable leading dims, wrong mask type or shape, a non-bool `qk_norm`, and `eps` given as bool/0/negative/NaN/string.

## Complexity

- Time \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S\cdot h_d)\): one matmul for the logits, \(O(L\cdot S)\) for activation and the row reduction, and another matmul against v.
- Apart from inputs and outputs, the weight matrix costs \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S)\); a tiled streaming normalization can bring that down to \(O(L\cdot h_d + S\cdot v_d)\), which is one of this attention family's engineering selling points over softmax.
