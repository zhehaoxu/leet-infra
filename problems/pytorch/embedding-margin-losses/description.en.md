# Embedding Margin Losses: CosineEmbeddingLoss and HingeEmbeddingLoss

## Problem

Besides triplets there are two **pairwise** metric-learning losses. CosineEmbeddingLoss (torch's `nn.CosineEmbeddingLoss`; its academic origin is usually traced to the cosine embedding loss of Jia et al., *Large-scale Video Classification with Convolutional Neural Networks*, CVPR 2014) looks only at the **angle** between two vectors: matching pairs must point the same way (\(1-\cos\)), non-matching pairs must be separated by more than a margin (\(\max(0,\cos-\text{margin})\)). It is therefore length-independent, which suits "project variable-length sentences/images onto the unit sphere". HingeEmbeddingLoss applies the standard SVM hinge to an **already-given pairwise score** (a positive pair uses the score directly as the loss; a negative pair must fall below the margin), as in Siamese networks with a scalar similarity head. This problem is a mode dispatcher that implements both losses plus all three reductions. CPU, fully deterministic, no random number calls.

## Rules

- The complete signature, argument meanings and defaults live only in the starter docstring; that is the sole judging contract. mode ∈ {`"cosine"`, `"hinge"`}, anything else raises ValueError; reduction ∈ {`"mean"`, `"sum"`, `"none"`}, anything else raises ValueError.
- **cosine branch**: x1, x2 of shape (N,D), target of shape (N,) with values in \(\{1,-1\}\) (not validated); `margin=None` means **0.0**; x2 is None raises ValueError. Let
  \[\cos_i=\frac{\langle x1_i,x2_i\rangle}{\max(\lVert x1_i\rVert_2\cdot\lVert x2_i\rVert_2,\;10^{-8})},\qquad
    \ell_i=\begin{cases}1-\cos_i,&\text{target}_i=1\\[2pt]\max(0,\ \cos_i-\text{margin}),&\text{target}_i=-1\end{cases}\]
  **the clamp applies to the product of the two norms as a whole** (not to each norm separately); if either vector is zero the denominator is floored at \(10^{-8}\) while the numerator is 0, so \(\cos_i=0\) and the target=1 branch is exactly 1.
- **hinge branch**: x1 is a shape-(N,) **pairwise score** (not a vector), target (N,) ∈ \(\{1,-1\}\); `margin=None` means **1.0**; x2 is expected to be None and is **ignored without validation**.
  \[\ell_i=\begin{cases}x1_i,&\text{target}_i=1\\[2pt]\max(0,\ \text{margin}-x1_i),&\text{target}_i=-1\end{cases}\]
  Note the target=1 branch is the **identity** (no relu), so a negative score yields a **negative loss** — this is the real behaviour of `F.hinge_embedding_loss`, pinned with `torch.equal` by the judge.
- **The two margins have different defaults** (cosine 0.0, hinge 1.0), both expressed as "pass None to get that branch's default", matching the native signatures.
- **N=0 convention**: `"none"` returns a (0,) empty tensor; `"sum"` returns 0; `"mean"` returns **scalar 0** (the native function yields NaN for an empty mean; this problem defines it as 0).
- Output dtype follows x1; shape (N,) for `"none"`, 0-dim scalar for `"mean"/"sum"`.

## Example

**cosine**: x1 \(=[[1,0,2],[0,1,1]]\), x2 \(=[[1,1,0],[0,-1,2]]\), target \(=[1,-1]\), default margin 0.0, reduction="none".
Both pairs have dot product 1 and norm product \(\sqrt5\cdot\sqrt2=\sqrt{10}\) → \(\cos=[0.3162278,\,0.3162278]\):

- i=0 (target=1): \(\ell=1-0.3162278=\) **0.6837722063**
- i=1 (target=−1): \(\ell=\max(0,\,0.3162278-0)=\) **0.3162277639**

**hinge**: x1 \(=[2.5,-0.5]\), target \(=[1,-1]\), default margin 1.0, reduction="none":

- i=0 (target=1, identity branch): **2.5**
- i=1 (target=−1): \(\max(0,\,1-(-0.5))=\) **1.5**

i.e. EXAMPLE_COSINE = [0.6837722063, 0.3162277639] and EXAMPLE_HINGE = [2.5, 1.5].

## Constraints

- mode accepts only `"cosine"`/`"hinge"`; reduction only `"mean"`/`"sum"`/`"none"`; x2=None under cosine, an illegal mode and an illegal reduction all raise ValueError
- target is only promised to lie in \(\{1,-1\}\) (a long tensor) and is not validated; other values fall into the `-1` branch (not judged)
- The denominator clamp threshold is fixed at \(10^{-8}\) and acts on the **product** of the norms; zero vectors and doubly-zero vectors must both produce finite values
- Cross-check: elementwise \(10^{-6}\) agreement with `F.cosine_embedding_loss` (cosine) and `F.hinge_embedding_loss` (hinge) over margin ∈ {0.0,0.2,0.7}/{1.0,0.5} × the three reductions; the shape/dtype of `"none"` versus `"mean"/"sum"` is itself judged
