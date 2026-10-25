# BatchNorm Forward (Training/Inference)

## Problem

Batch Normalization normalizes along the channel dimension (dim=1) across the batch (and spatial dimensions), and its behavior differs between training and inference:

- **Training**: normalize with the current batch's statistics and move the statistics into the running stats by an exponential moving average:
  $$ \mu_B,\ \sigma_B^2 = \text{mean/var over dims } (0,2,3,\dots),\quad y = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}\gamma + \beta $$
  $$ \text{running\_mean} \leftarrow (1-m)\cdot\text{running\_mean} + m\cdot\mu_B,\qquad \text{running\_var} \leftarrow (1-m)\cdot\text{running\_var} + m\cdot\hat\sigma^2 $$
- **Inference**: normalize directly with the running stats, and do not update them.

## Rules

- Implement `solve(x, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5)`, returning the triple `(y, running_mean_new, running_var_new)`.
- The input is `(N, C)` or `(N, C, *)`; the statistics are taken over all dimensions except `dim=1`; $\gamma,\beta$ broadcast per channel.
- **Two classic contract traps** (exactly matching `F.batch_norm`):
  1. The **normalization denominator** uses the **biased** variance (divide by $N$);
  2. The **running_var update** uses the **unbiased** variance (divide by $N-1$, i.e. $\hat\sigma^2 = \frac{n}{n-1}\sigma_B^2$, where $n$ is the number of elements per channel).
- Do not modify the passed-in running tensors in place; return the updated new values.

## Examples

```
Input: x = [[1,2],[3,4]] (N=2,C=2), γ=[1,1], β=[0,0],
      running_mean=[0,0], running_var=[1,1], training=True, momentum=0.1
batch statistics: μ=[2,3], biased variance=[1,1], unbiased variance=[2,2]
Output: y=[[≈-1.0, ≈-1.0],[≈1.0, ≈1.0]]        # actually ±0.99999 (eps=1e-5)
      running_mean_new=[0.2, 0.3]
      running_var_new=[1.1, 1.1]              # 0.9*1 + 0.1*2 (unbiased variance!)
```

## Constraints

- float32/float64 input; the number of statistics elements per channel is $n \ge 2$
- The result must match `torch.nn.functional.batch_norm`
