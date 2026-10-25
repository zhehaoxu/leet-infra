# Soft Dice Loss

## Problem

Segmentation tasks commonly use the **Dice coefficient** to measure the overlap between the predicted mask and the ground-truth mask: \(\text{Dice}=\frac{2|X\cap Y|}{|X|+|Y|}\). It is friendlier than per-pixel cross-entropy for the problem of insensitivity to small objects/positive samples. The **Soft Dice loss** replaces the hard 0/1 mask with a continuous probability \(p\in[0,1]\), making it differentiable:

\[ \mathcal{L}_{\text{dice}} = 1 - \frac{2\sum_c p_c y_c + s}{\sum_c p_c + \sum_c y_c + s} \]

The intersection-over-union relation between numerator and denominator comes directly from V-Net / MONAI `DiceLoss`. The smoothing term \(s\) prevents division by all-zero.

## Rules

Implement `solve(prob, target, smoothing=1.0, reduction="mean")`. `prob` (float32, \((N, C)\), predicted probabilities in \([0,1]\)) and `target` (float32, \((N, C)\), soft/hard labels in \(\{0,1\}\)). **Per sample**, along the last dimension \(C\), compute: \(\text{inter}=\sum_c p_c y_c\), \(\text{denom}=\sum_c p_c+\sum_c y_c\), \(loss_i = 1-\frac{2\,\text{inter}+s}{\text{denom}+s}\). Reduce according to `reduction` (`"mean"`/`"sum"`/`"none"`). An illegal reduction raises `ValueError`.

## Examples

\(N=1, C=2\), `prob=[[0.9,0.1]]`, `target=[[1,0]]`, `smoothing=1.0`:

```
inter = 0.9*1 + 0.1*0 = 0.9
denom = (0.9+0.1) + (1+0) = 2.0
dice  = (2*0.9 + 1) / (2.0 + 1) = 2.8/3 = 0.933333
loss  = 1 - 0.933333 = 0.0666667
```

## Constraints

- CPU float32; `prob` is already in \([0,1]\) (normalization along C is not required), `target` is 0/1.
- Summation is per sample along the **last dimension**, then reduced over the batch.
- `smoothing` (\(s\)) is added to both numerator and denominator to prevent division by zero.
- With a perfect prediction \(p=y\) (and not all zero), dice=1 and loss=0; all-zero samples are covered by \(s\), giving loss=0.
