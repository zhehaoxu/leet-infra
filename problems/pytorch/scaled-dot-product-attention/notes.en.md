# Scaled Dot-Product Attention Explained

## Formula

$$ \text{Attn}(Q,K,V)=\text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}+\text{mask}\Big)V $$

It is the minimal kernel of all attention mechanisms; Transformers, cross-attention, and FlashAttention all add structure on top of it.

## Why Divide by $\sqrt{d_k}$ (the heart of this problem)

Assume the components of $q,k$ are independent with mean 0 and variance 1; then the dot product $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ has variance $d_k$. Without scaling, the larger $d_k$, the larger the logits' variance, and softmax falls into the **saturation region** (nearly one-hot) with gradients approaching 0. Dividing by $\sqrt{d_k}$ pulls the logits' variance back to 1, keeping softmax in a regime with healthy gradients.

## Mask semantics: torch-native True = keep (the big trap here)

- PyTorch's native `bool mask`: **True means allowed to attend (keep)**, False means blocked. Hand-written code is often misled by intuition into "True = mask out" — the direction is exactly reversed.
- Practical implementation: `scores.masked_fill(~mask, -inf)` then add and apply softmax.
- When an **entire row is masked**, all entries are $-\infty$ → softmax gives `NaN`; torch's convention is that **the row outputs 0**, so use `nan_to_num(..., nan=0.0)`.

## Numerical Stability

Softmax must always subtract the row maximum before `exp` (see the "Softmax" problem): `scores - scores.max(-1,keepdim)` then `exp` and normalize, avoiding overflow on large logits.

## Dimensions

$Q:(...,L_q,d_k)$, $K:(...,L_k,d_k)$, $V:(...,L_k,d_v)$; $QK^\top:(...,L_q,L_k)$, multiplying $V$ gives $(...,L_q,d_v)$. $d_k$ appears only in $Q,K$, while $d_v$ is in $V$ — the scaling uses only $d_k$.

## Reference for Cross-Checking

Consistent with `F.scaled_dot_product_attention` (passing `attn_mask`); note the latter's mask also uses additive/boolean keep semantics.
