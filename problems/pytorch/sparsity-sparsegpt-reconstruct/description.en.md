# Sparsification: SparseGPT Column-Wise Pruning with Hessian Error Compensation

## Problem

One-shot magnitude pruning moves the **output** away from the original model: the removed weights vanish, and so does their contribution to the next layer's activations. The SparseGPT / OBS family fixes this by spreading each removed element's **error** onto the columns **not yet processed**, using second-order information (the inverse Hessian) from calibration data — keeping a hard mask while minimising the layer's output error.

Formally, minimise \(\|W X-\hat W X\|_2^2\) subject to \(\hat W\) obeying the mask. The OBS solution scans column by column: at column \(j\),

$$
Q_j = W_{:,j}\odot m_{:,j},\qquad
e_j=\frac{W_{:,j}-Q_j}{[H^{-1}]_{jj}},\qquad
W_{:,j}=Q_j,\qquad
W_{:,j+1:}\mathrel{-}=e_j\otimes [H^{-1}]_{j,\,j+1:}
$$

where \(H=XX^\top\) is the calibration second moment and \(H^{-1}\) is supplied from outside (real implementations invert it incrementally with Cholesky; this problem hands it to you and focuses on the compensation rule). Once column \(j\) is fixed by its mask, its error is pushed only to the right; each right-hand column applies its own mask when its turn comes and keeps propagating.

## Rules

Given `weight` (float32, \((R,C)\)), `mask` (float32, \((R,C)\), 0 means pruned) and `Hinv` (float32, \((C,C)\), symmetric positive definite, i.e. \(H^{-1}\)):

for \(j=0,1,\dots,C-1\) in order:

1. `cur = W[:, j]` (current values, possibly already nudged by earlier columns);
2. `Q = cur * mask[:, j]`;
3. `e = (cur - Q) / Hinv[j, j]`;
4. `W[:, j] = Q`;
5. if \(j+1<C\): `W[:, j+1:] -= e ⊗ Hinv[j, j+1:]` (outer product).

Return `W_out`, float32, shape \((R,C)\). `torch.linalg.inv` and existing SparseGPT implementations are forbidden.

## Examples

```
weight = [[1, 2, 3],
          [4, 5, 6]]
mask   = [[1, 0, 1],
          [0, 1, 1]]
Hinv   = [[2, 1, 0],
          [1, 2, 1],
          [0, 1, 2]]

j=0: cur=[1,4]  Q=[1,0]   e=([1,4]-[1,0])/2=[0,2]
     W[:,0]=[1,0]; W[:,1:] -= [0,2]⊗[1,0] -> col1 [2,5]-[0,2]=[2,3], col2 unchanged [3,6]
j=1: cur=[2,3]  Q=[0,3]   e=([2,3]-[0,3])/2=[1,0]
     W[:,1]=[0,3]; W[:,2] -= [1,0]⊗[1] = [3,6]-[1,0] = [2,6]
j=2: cur=[2,6]  Q=[2,6]   e=0, no columns to the right

W_out = [[1, 0, 2],
         [0, 3, 6]]
```

## Constraints

- CPU float32; `Hinv` is symmetric positive definite (its diagonal may be small but is never 0).
- Masked positions must be **exactly 0**; unmasked positions may be rewritten by later compensation and are not guaranteed to keep their original values.
- The last column has nothing to the right, so its error is discarded in place.
- With `Hinv` equal to the identity there is no cross-column coupling and the result must be exactly the elementwise mask `weight * mask`.
