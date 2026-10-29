# Notes: Dimension Bookkeeping in the Linear Backward

## Origin

Any role that involves writing custom `autograd.Function` objects (fused MLP, quantized linear, LoRA branches) starts with this question: given dY, how do you write dX and dW. What separates candidates is not the formula but why dW must sum over the batch, and at which step that summation happens under multi-GPU or multi-layer setups.

## Why dW Must Carry a Folded Sum

Write \(Y_{b,m,n}=\sum_k X_{b,m,k}W_{n,k}\). Then

$$\frac{\partial L}{\partial W_{n,k}}=\sum_b\sum_m \frac{\partial L}{\partial Y_{b,m,n}}\,X_{b,m,k}$$

because \(W\) is shared by every batch element and every position, so the gradient accumulates over **all axes except (N, K)**. einsum is the least error-prone spelling: `dW = einsum('...mn,...mk->nk', dY, X)`. Reshaping to 2-D first silently misaligns whenever \(B\) and \(M\) cannot be commuted on non-contiguous memory. dX has no reduction at all: `einsum('...mn,nk->...mk', dY, W)`.

## Core Ideas and Pitfalls

- **Forgetting the fold**: `dY.T @ X` on 3-D input either errors or yields \(B\) separate `(N,K)` blocks instead of one.
- **Which transpose**: `W` is `(out_features, in_features)` and the forward uses `W.T`, hence dX multiplies by `W` while dW is `dY.T @ X`; the two transposes are easy to swap.
- **dtype ownership**: dX follows X and dW follows W; returning both in float32 breaks the entry-wise comparison with autograd under mixed precision.
- **Zero sizes**: with `M=0` the matmul gives all zeros rather than an empty tensor, matching autograd — do not return early.

## How It Differs from Neighboring Problems

`ag-attention-qkv-backward` is also a matmul backward but its three gradients are coupled through a shared dP; `linear-layer` is forward-only and never touches a reduction. This problem stays single-operator on purpose, concentrating the difficulty on batch folding and dtype ownership.

## Cross-Check Basis and Complexity

The oracle uses `torch.autograd`: mark X and W as requires_grad, form `Y=X@W.T`, `loss=(Y*dY).sum()` and take grads. Complexity is \(O(BMKN)\) (the folded sum is the same order as the two products) plus an \(O(NK)\) accumulation buffer.
