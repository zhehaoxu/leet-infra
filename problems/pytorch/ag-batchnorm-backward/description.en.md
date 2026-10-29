# BatchNorm Training Backward: the Three Terms and the Double Counting

## Problem Description

In training mode BatchNorm carries **differentiable statistics**: \(\mu,\sigma\) are computed from the current batch, so each \(x_{n,c}\) affects the output directly through \(x̂_{n,c}\) and also, via the \(\mu_c,\sigma_c\) it helps build, affects **every** sample in that column. The backward therefore splits naturally into three parts — the pass-through term, the mean-subtraction term and the std-division term — and writing the \(1/N\) factor of the last two as \(1/N^2\) (or counting it twice) is precisely this problem's famous double-counting bug. Given `x (N,C), gamma, beta, dY`, return `(dX, dGamma, dBeta)`.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- Reductions run along **axis 0 (the batch axis)**, per feature, with the biased variance (`unbiased=False`).
- With \(a=dY\cdot\gamma\): \(dX=\big(a-\text{mean}_N(a)-\hat x\cdot\text{mean}_N(a\circ\hat x)\big)/\sigma\), all three means along the batch axis.
- \(d\gamma=\sum_N dY\circ\hat x\) and \(d\beta=\sum_N dY\), of shape `(C,)`.
- Use the statistics of **this very batch** (training mode); `running_mean/running_var` must not appear. A negative `eps` or a shape mismatch raises ValueError.

## Example

4 samples, 2 features, `eps=0.0`; numbers come from the EXAMPLE_* constants in test_cases.py:

```
x      = [[-1,  1],     gamma = [2, 1]  beta = [0, 0]   dY = [[1, 2],
          [ 1, -1],                                          [3, 4],
         [-1,  1],                                           [5, 6],
          [ 1, -1]]                                          [7, 8]]
mu = [0, 0], std = [1, 1], a = dY*gamma = [[2, 2], [6, 4], [10, 6], [14, 8]]
mean_N(a) = [8, 5], mean_N(a*xhat) = [2, -1]
dX     = [[-4, -2], [-4, -2], [4, 2], [4, 2]]
dGamma = [4, -4]     dBeta = [16, 20]
```

Column 0, row 0: \(2-8-(-1)\cdot2=-4\). Because \(\hat x\) only takes \(\pm1\), dX can assume only two distinct values within a column.

## Constraints

- `x` is a 2-D float32 or float64 tensor; accumulate in float64 and return the dtype of `x`
- Cross-checked against `torch.autograd.grad` of `torch.nn.functional.batch_norm(train=True)`, atol=1e-9 (float64) and 1e-5 (float32)
- Self-check: with `eps=0`, \(\sum_N dX=0\) and \(\sum_N dX\circ\hat x=0\) must hold exactly
