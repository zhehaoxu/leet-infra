# LoRA Low-Rank Adaptation Forward Explained

## Source
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685). Core assumption: during fine-tuning the weight update \(\Delta W\) has low "intrinsic rank", so parameterize \(\Delta W = B A\) (\(B:d_\text{out}\times r\), \(A:r\times d_\text{in}\)), freeze \(W_0\) during training and update only \(A,B\). The paper takes the scaling factor as \(\alpha/r\) (the original often uses \(\alpha=2r\) when \(r\) is fixed so the coefficient is constant, but engineering practice generally scales explicitly by \(\alpha/r\)). Initialization \(A\sim\mathcal N(0,\sigma^2)\), \(B=0\), so at the training start \(\Delta W=0\) and the pretrained model is not damaged.

## Why it is written this way (pitfalls)
1. **The scale \(s=\alpha/r\) multiplies only the side path**: \(y_0=xW_0^\top\) does not contain \(s\); only the low-rank term \(y_\Delta=s\,(xA^\top)B^\top\) is multiplied by \(s\). Forgetting \(s\), or also multiplying the base by it, is wrong.
2. **Matrix order**: \(\Delta W = B A\) (\(A\) down first, then \(B\) up), shape \((d_\text{out},d_\text{in})\). Writing \(A B\) gives the wrong shape.
3. **Row-vector convention**: this collection follows `nn.Linear`'s \(y=xW^\top\). Hence the side path is \((x A^\top) B^\top\), not \(x A B\).
4. **Merge equivalence**: \(y = x(W_0+\frac{\alpha}{r}BA)^\top\) equals the step-by-step sum element-wise — this is the basis for "LoRA can be folded in, with zero latency overhead at deployment", and also the structural difference in this problem's oracle (the oracle merges the matrices, while the reference solution uses two separate matrix multiplications).

## Grading details
The reference solution uses "base + step-by-step side path" in two matrix multiplications; the grader's `_oracle` uses "first merge \(W'=W_0+sBA\), then one \(xW'^\top\)" — a different structure but mathematically equivalent, compared with `allclose(atol=1e-5,rtol=1e-5)`. Checks cover: the example values, random-weight comparison, `alpha=0` degenerating to the pure base, 3D batches, and equivalence to the merged matrix.

## Complexity
The forward pass adds \(O(n\cdot r\cdot(d_\text{in}+d_\text{out}))\) (far less than the dense \(\Delta W\)'s \(O(n\,d_\text{in}d_\text{out})\)); trainable parameters \(r(d_\text{in}+d_\text{out})\).
