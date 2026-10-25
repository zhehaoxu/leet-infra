# IA3 Element-wise Activation Rescaling Explained

## Source
Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (arXiv:2205.05638, (IA)³). The paper multiplies a learnable vector onto the K and V of self-attention and onto the FFN intermediate activations, freezing everything else. This problem abstracts it to rescaling the output of a single linear layer as a whole, capturing the core operator "multiplicatively scaling activations".

## Core points and pitfalls
1. **Acts on the output dimension**: \(\lambda\in\mathbb R^{d_\text{out}}\), \(y=\lambda\odot(xW^\top)\), i.e. it broadcasts on the last dimension; it does not act on the input dimension.
2. **Broadcasting must span batch and sequence**: for \((b,l,d_\text{out})\) it also multiplies the same \(\lambda\) along the last dimension. `h * lam` (automatic broadcasting) suffices; if you write a manual loop, do not align `lam` to `d_in`.
3. **Identity initialization**: \(\lambda=\mathbf 1\) leaves the pretrained model unchanged; \(\lambda=0\) zeros that output dimension directly (suppression).
4. **Equivalence to folding the weights**: \(\lambda\odot(xW^\top) = x(\lambda_\text{row}\odot W)^\top\) (multiply \(\lambda\) into each row of \(W\)). Hence IA³ can also be folded into the weights for deployment.

## Grading details
The reference solution first computes \(h=xW^\top\) and then multiplies by \(\lambda\); the grader's `_oracle` takes the path "multiply \(\lambda\) into the weight rows to get \(W'=\lambda[:,\text{None}]\odot W\), then one \(xW'^\top\)" — a different structure, mathematically equivalent, `allclose(atol=1e-5,rtol=1e-5)`. Additional cases: \(\lambda=\mathbf1\) identity, \(\lambda=0\) all zeros, and 3D batch assertions.

## Complexity
Each layer adds only \(d_\text{out}\) trainable parameters (fewer than LoRA's \(r(d_\text{in}+d_\text{out})\)); the forward pass adds one element-wise multiply \(O(n\,d_\text{out})\).
