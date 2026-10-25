# In Depth: Soft Dice Loss

## Origin
The Dice coefficient originates from the Sørensen–Dice similarity; the soft/differentiable Dice loss appears in Milletari et al., *V-Net* (3D medical segmentation, arXiv:1604.03265), MONAI `DiceLoss`, and nnU-Net. Two denominators are common: linear \(\sum p+\sum y\) (this problem) and squared \(\sum p^2+\sum y^2\) (the Dice Loss squared variant). The problem statement explicitly uses the linear form with smoothing \(s\).

## Core points and pitfalls
1. **Aggregate along the channel dimension, not element-wise**: Dice is a set overlap, so first sum each sample's \(C\) dimension into intersection/size terms, then compute the ratio; doing it element-wise degenerates to a constant.
2. **The denominator is \(|p|+|y|\), not the union**: \(\frac{2\sum py}{\sum p+\sum y}\) is soft Dice for continuous probabilities; don't write IoU's union denominator.
3. **The smoothing term \(s\) is added to both numerator and denominator**: all-zero samples stay NaN-free and the loss goes to 0.
4. **Per-sample reduction**: first produce one loss per row, then reduce over the batch according to `reduction`; do not mix the whole batch into a single Dice (unless the problem asks for dice mode).

## Verification approach
The reference solution uses `inter=(p*y).sum(-1)`, `denom=p.sum(-1)+y.sum(-1)`, `loss=1-(2inter+s)/(denom+s)`, then reduces according to `reduction`. The judge's `_oracle` accumulates numerator and denominator per sample in pure Python (different structure, numerically equivalent). Additional assertions cover "perfect prediction loss=0", "all-zero sample loss=0 (smoothing fallback)", "completely opposite predictions, prob=[[1,0]] vs y=[[0,1]], give loss≈1", and "reduction none/sum/mean are consistent".

## Complexity
\(O(N C)\).
