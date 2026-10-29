# Notes: The Two Dual Paths of the Convolution Backward

## Origin

The `conv2d` backward was a mandatory question for CNN-era framework roles and remains the standard sample for "write an operator backward by hand", because it compresses three easily confused concepts into two formulas: correlation versus convolution, the gradient boundary of padding, and the batch/spatial folding of \(dw\). Vision interviewers may push on im2col or Winograd; this problem covers the most basic and most error-prone part.

## Why dx Is a Correlation with the Flipped Kernel

The forward is a correlation: \(y_u=\sum_k w_k x_{u+k}\) (channel and 2-D details omitted). Then \(dx_j=\sum_k dY_{j-k}w_k\), which is precisely the definition of correlating with a **flipped** kernel. Two equivalent implementations exist: `conv_transpose2d` with `weight.flip(-1,-2)`, or zero-padded ordinary convolution of \(dY\) with the flipped kernel using \(k_h-p-1\) padding. Forgetting the flip is invisible for symmetric kernels (such as the all-ones \(2\times2\) in the example) — which is exactly why the test also ships asymmetric random kernels.

## The Summation Range of dw

\(dw[co,ci,k,l]=\sum_{b,u,v}dY[b,co,u,v]\,x[b,ci,us+k-p,vs+l-p]\), skipping any \((u,v)\) whose patch leaves \(x\). This is the same fact seen from the other side as "dx discards out-of-range contributions": padding shows up as missing terms in \(dw\) and as rows whose gradient never reaches the input in \(dx\). The folded axes must include the batch, the same lesson as dW in `ag-matmul-backward`.

## Core Ideas and Pitfalls

- **Interleaved zeros for `stride>1`**: \(dx\) equals convolving \(dY\) after inserting `stride-1` gaps; skipping the insertion shrinks every spatial dimension.
- **Shape of `dx` under `padding>0`**: it stays the shape of `x`, not the de-padded one.
- **Channels**: \(dx\) sums over `Cout` while \(dw\) sums over `B/H/W`, keeping `Cin` and `Cout` intact — mixing up even one of the four axes breaks everything.
- **Numbers**: the example sums \(1..9\) with an all-ones kernel so every value is an integer and judging can use `atol=1e-9`.

## How It Differs from Neighboring Problems

`ag-pooling-backward` has no learnable parameters and only routes gradients back into a window; `ag-matmul-backward` folds a single spatial axis. The signature difficulty here is that one and the same \(dY\) must be correlated once with a flipped kernel and once without.

## Cross-Check Basis and Complexity

The oracle runs `F.conv2d(x, weight, stride=..., padding=...)` and takes `torch.autograd.grad((y*dY).sum(), [x, weight])`, atol=1e-9 in float64. Complexity is \(O(B\,C_{in}C_{out}HWH^2)\), with \(dx\) and \(dw\) of the same order.
