# Cosine Similarity in Depth

## Direction, not magnitude

The dot product $a\cdot b$ depends on both the **angle** and the **length**; dividing by the two norms normalizes the lengths away and leaves only the angle information:
$$ \cos = \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert} = \hat a\cdot \hat b $$
Hence the similarity of $a$ and $7a$ is 1 (same direction), orthogonal vectors give 0, and opposite vectors give $-1$. Embedding retrieval favors it
precisely because "saying the same concept more or less" should be judged as similar.

## Relation to L2 distance and inner product

For **normalized** vectors, $\lVert \hat a - \hat b\rVert^2 = 2 - 2\cos$, so "maximizing cosine"
is equivalent to "minimizing spherical L2". Without normalization, the dot product also counts the magnitudes and is often used for weighted importance scoring;
which one to choose depends on "whether length carries semantics".

## Two ways to write eps

- `clamp_min(eps)` (PyTorch / this problem): the norm is floored only when it is smaller than eps, so normal values are unaffected.
- `+eps` (common in paper pseudocode): eps is added to every denominator, slightly lowering the similarity at small angles.
Both are safe for zero vectors, but the values differ — make sure to use the same one as the reference implementation `F.cosine_similarity`.

## In contrastive learning

InfoNCE/NT-Xent divides the similarities between one positive sample and several negative samples by the temperature and then applies softmax: the positive sample should be maximized
and the negatives minimized. So "cosine → /τ → softmax" is the skeleton of the contrastive loss; understanding this problem
is the first step toward understanding the SimCLR, MoCo, and CLIP losses.
