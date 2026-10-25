# Batched Matrix Multiplication (Batched GEMM) Explained

## Definition

`torch.matmul`'s batching semantics: for high-dimensional (>2D) tensors, treat the **last two dimensions as a matrix** and all leading dimensions as batch dimensions to broadcast:

$$ (\dots,m,k)\ @\ (\dots,k,p)\ \to\ (\dots,m,p) $$

The 1D cases have promote/demote rules (1D@1D→scalar, 1D@2D→prepend row/column); this problem focuses on ≥2D.

## Broadcasting rules (the focus of this problem)

Batch dimensions align by **ordinary broadcasting**: `torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])`. Hence:

- $(2,3,4)@(2,4,5)\to(2,3,5)$
- $(3,4,5)@(1,5,6)\to(3,4,6)$ (batch dimension $1$ broadcasts to $3$)
- a single matrix $(m,k)@(k,p)\to(m,p)$ (empty batch)

The inner dimensions must be equal ($a_{\text{last}}=b_{\text{second-last}}$), otherwise it errors — this is the only hard constraint.

## Element-wise implementation (the form used by this reference solution)

Without relying on `@`, use the outer-product-sum definition of matrix multiplication:

$$ C_{ij}=\sum_k A_{ik}B_{kj}\ \Rightarrow\ (a.\text{unsqueeze}(-1)*b.\text{unsqueeze}(-3)).\text{sum}(-2) $$

First `expand` the batch to the common shape, then compute $a(\dots,m,k,1)\times b(\dots,1,k,p)$ and sum along $k$.

## Why practice it separately

All attention, FFN, and backward gradient computations are built on batched matmul; `einsum('...ik,...kj->...ij')` is its readable equivalent. Understanding broadcasting and the inner-dimension constraint is the basis for understanding the difference between `bmm/matmul` (bmm strictly requires 3D and does not broadcast the batch).
