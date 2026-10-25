# Cosine Similarity

## Problem

Cosine similarity measures how close two vectors are in **direction**, independent of their lengths:

$$ \cos(a,b) = \frac{a\cdot b}{\lVert a\rVert_2\,\lVert b\rVert_2} \in [-1, 1] $$

It is the default metric in retrieval/recommendation (embedding recall), contrastive learning (scoring positive and negative samples in InfoNCE/SimCLR), and
semantic similarity (sentence-BERT). Implement it and handle division-by-zero protection for zero vectors correctly.

## Rules

- Implement `solve(a, b, dim=-1, eps=1e-8)`, where `a` and `b` have the same shape (or are broadcastable).
- Compute the dot product and the norms along `dim`; the output shape is the shape with that dimension removed.
- Protect the denominator with `clamp(min=eps)` (consistent with `F.cosine_similarity`) to avoid division by zero for zero vectors.
- The result must match `F.cosine_similarity(a, b, dim=dim, eps=eps)`.

## Examples

```
Input: a = [1, 2, 3], b = [4, 5, 6]
  a·b = 4+10+18 = 32
  ‖a‖ = √14 ≈ 3.7417, ‖b‖ = √77 ≈ 8.7749
  cos = 32 / (3.7417 × 8.7749) ≈ 0.97463
Output: [0.97463]
```

## Constraints

- float32 input; the output lies in `[-1, 1]`
- The similarity of a vector with itself should be 1 (scaling by a positive multiple does not change direction)
