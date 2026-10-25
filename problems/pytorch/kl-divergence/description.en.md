# KL Divergence (Forward and Reverse)

## Problem

KL divergence (Kullback–Leibler divergence) measures the extra information cost of approximating a true distribution \(P\) with a distribution \(Q\). It is **asymmetric**: \(\mathrm{KL}(P\|Q)\ne\mathrm{KL}(Q\|P)\). Forward KL (mean-seeking) is common in variational inference/MLE, while reverse KL (mode-seeking) is preferred in knowledge distillation—which is precisely why the distillation loss is written as \(\mathrm{KL}(\text{teacher}\|\text{student})\).

\[ \mathrm{KL}(P\|Q) = \sum_i P_i \log\frac{P_i}{Q_i},\qquad 0\log 0 := 0 \]

## Rules

Implement `solve(P, Q)`, returning a 2-tuple `(forward_kl, reverse_kl)`, where `forward_kl = KL(P‖Q)` and `reverse_kl = KL(Q‖P)`. `P` and `Q` are float32, shape \((N, C)\), **row-normalized** probability distributions (each row sums to 1). For each row, sum over \(c\) by definition \(\sum_c P_{c}\log(P_c/Q_c)\) (with the convention \(0\log 0=0\), and \(Q_c>0\) whenever \(P_c>0\)), then take the **mean** over the \(N\) rows to obtain a 0-dimensional scalar. Both return values are scalar tensors.

## Examples

\(N=2, C=2\), `P=[[0.7,0.3],[0.2,0.8]]`, `Q=[[0.4,0.6],[0.5,0.5]]`:

```
forward_kl = KL(P‖Q) = mean([0.7·ln(0.7/0.4)+0.3·ln(0.3/0.6),
                             0.2·ln(0.2/0.5)+0.8·ln(0.8/0.5)]) = 0.18826580
reverse_kl = KL(Q‖P) = 0.20759282
```

## Constraints

- CPU float32; each row of `P` and `Q` is a normalized probability distribution.
- Convention \(0\log 0 = 0\) (i.e. terms with \(P_c=0\) contribute 0).
- The return value must be two 0-dimensional tensors `(forward_kl, reverse_kl)`; they are generally unequal (asymmetry).
- When `P` and `Q` are identical, both return values should be approximately 0.
