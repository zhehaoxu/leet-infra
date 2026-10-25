# Multimodal: CLIP Symmetric InfoNCE Contrastive Loss

## Problem

CLIP (Radford et al., arXiv:2103.00020) aligns the image and text encoders with a **symmetric InfoNCE** loss: take the \(N\) image features and \(N\) text features in a batch to form an \(N\times N\) similarity matrix, whose diagonal holds the \(N\) positive pairs and whose off-diagonal entries are negatives. Run one "pick out the correct pairing" cross-entropy in each direction and average them:

\[ \text{logits}_{i2t} = s\,(I T^\top),\quad \text{logits}_{t2i} = s\,(T I^\top),\quad y=\mathrm{arange}(N) \]
\[ \mathcal{L} = \tfrac12\bigl[\mathrm{CE}(\text{logits}_{i2t},y) + \mathrm{CE}(\text{logits}_{t2i},y)\bigr] \]

where \(s\) is the learnable temperature scale `logit_scale`.

## Rules

Implement `solve(image_feats, text_feats, logit_scale)`. Both feature tensors are already L2-normalized float32 of shape \((N,D)\); `logit_scale` is a scalar. Return a 0-dimensional float32 loss scalar according to the formula above. `CE` is the row-wise softmax cross-entropy (equivalent to `F.cross_entropy`); calling contrastive-learning libraries directly is forbidden.

## Examples

\(N=2,D=2\): `image=[[1,0],[0.6,0.8]]`, `text=[[1,0],[0.2,0.98]]`, `logit_scale=4.0`:

```
logits_i2t = 4 * image@text.T ; labels=[0,1]
loss = 0.5*(CE(logits_i2t,[0,1]) + CE(logits_t2i,[0,1])) = 0.135395
```

## Constraints

- CPU float32; input features are assumed already normalized (this problem does not normalize them).
- Symmetric = cross-entropy along the rows of both the i2t and t2i directions, then averaged.
- The larger the temperature \(s\) → the sharper the softmax (the more it penalizes negatives).
