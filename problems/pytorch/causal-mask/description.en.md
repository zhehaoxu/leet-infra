# Causal Mask

## Problem

When an autoregressive model (GPT-style) computes attention, position $i$ may only see tokens with index $\le i$ and must not "peek at the future". This is done by keeping the lower triangle (including the diagonal) of the attention score matrix and setting the upper triangle to $-\infty$, so that after softmax the probabilities of future positions are 0.

$$ \text{masked}[i,j] = \begin{cases} \text{scores}[i,j] & j \le i \\ -\infty & j > i \end{cases} $$

## Rules

- Implement `solve(scores)`, where the input is attention scores of shape `(..., L, L)`, and return a tensor of the same shape with the upper triangle set to $-\infty$.
- What is kept is the **lower triangle including the diagonal** (`tril`): the first $i+1$ elements of row $i$ stay unchanged and the rest become $-\infty$.
- Arbitrary leading batch dimensions are supported; the mask depends only on the last two dimensions and the same triangular structure must broadcast across all batches/heads.
- Mask the scores directly (this problem does not involve softmax), and you must use `float("-inf")` rather than a finite large negative number.

## Examples

```
Input (4x4, value = i*4+j):
[[ 0, 1, 2, 3],
 [ 4, 5, 6, 7],
 [ 8, 9,10,11],
 [12,13,14,15]]
Output:
[[ 0, -inf, -inf, -inf],
 [ 4,   5 , -inf, -inf],
 [ 8,   9 ,  10 , -inf],
 [12,  13 ,  14 ,  15 ]]
```

## Constraints

- float32/float64 input
- The last two dimensions are equal ($L \times L$)
