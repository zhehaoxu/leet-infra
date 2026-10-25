# Sparsity: 2:4 Structured N:M Pruning

## Problem

**2:4 structured sparsity** is the "semi-structured" pattern natively supported by the NVIDIA Ampere/Hopper Sparse Tensor Core: split the weights along the last dimension into **groups of 4 consecutive elements**, and in each group **keep only the 2 with the largest absolute value**, zeroing the rest ("out of every four, two must be zero"). It combines accuracy close to unstructured pruning with a regular memory layout that hardware can accelerate, which is why it is a frequently asked topic in LLM inference pruning.

## Rules

Input `weight` (float32, last dimension length divisible by 4). Cut it into groups with stride 4 along the last dimension; inside each group keep the 2 elements with the largest absolute value (original values and signs unchanged) and zero the other 2. On a tie (equal \(|w|\) within a group) keep the **earlier index**. Return a float32 tensor of the same shape. `torch.sparse`/cuSPARSELt are not allowed.

## Examples

`weight=[[1,-5,3,-2, 0.5,-9,2,8]]` (shape 1×8, two groups):

```
Group 0 [1,-5,3,-2]: |w|=[1,5,3,2], top-2 = -5, 3 -> [0,-5,3,0]
Group 1 [0.5,-9,2,8]: |w|=[0.5,9,2,8], top-2 = -9, 8 -> [0,-9,0,8]
out = [[0,-5,3,0, 0,-9,0,8]]
```

## Constraints

- CPU float32; the last dimension is divisible by 4; grouping happens only within the last dimension (different rows / different groups do not affect each other).
- Each group keeps exactly 2 and prunes 2, so the overall sparsity ratio is always 50%.
- The sign of a retained element is irrelevant; only absolute values are compared.
