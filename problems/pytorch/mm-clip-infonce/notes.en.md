# In Detail: CLIP Symmetric InfoNCE Contrastive Loss

## Source
Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, arXiv:2103.00020): symmetric InfoNCE / NT-Xent, which runs a cross-entropy with the diagonal as the positive sample in both the image→text and text→image directions and then averages them, with a learnable `logit_scale` as the temperature.

## Core points and common mistakes
1. **Symmetric = compute CE in both directions**: \(\tfrac12[\mathrm{CE}(s\,IT^\top,y)+\mathrm{CE}(s\,TI^\top,y)]\). Computing only one direction underestimates the loss and is asymmetric.
2. **labels is always arange(N)**: the diagonal holds the positive pairs, so the correct class for row \(i\) is column \(i\).
3. **The temperature multiplies the logits** (`logit_scale` is usually ≈exp(learned)≈1/τ); the larger it is, the sharper the softmax. Do not apply the scale outside the loss.
4. Cross-entropy contains a logsumexp internally, so `F.cross_entropy` guarantees numerical stability.

## Cross-check convention
The reference solution is `0.5*(F.cross_entropy(scale*I@T.T, arange)+ the same for t2i)`. The grading `_oracle` computes the softmax NLL by hand in pure Python with a per-row max-shift + `logsumexp`, averages each of the two directions, then averages the two (different structure, float64 accumulation, numerically equivalent), with `abs(got-exp)<1e-5`. It also asserts "identical features + large temperature → loss approaches 0", "the scalar is symmetric", and "identity similarity".

## Complexity
Matrix multiplication \(O(N^2 D)\) + softmax \(O(N^2)\).
