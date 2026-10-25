# Jensen-Shannon Divergence Loss

## Problem

KL divergence is asymmetric and does not satisfy the triangle inequality, so using it directly as a symmetric distribution-matching loss is not ideal. **Jensen–Shannon divergence (JSD)** symmetrizes the two KLs to obtain a bounded, symmetric dissimilarity, widely used in pseudo-labeling/self-training, distribution alignment, and consistency regularization for image segmentation.

\[ \mathrm{JSD}_\beta(P\|Q) = \beta\,\mathrm{KL}(P\|M) + (1-\beta)\,\mathrm{KL}(Q\|M),\quad M = \beta P + (1-\beta) Q \]

Here \(\beta\in(0,1)\) is the mixing weight (commonly \(0.5\)). When \(\beta=0.5\), JSD is symmetric in \(P,Q\) and is bounded above by \(\ln 2\).

## Rules

Implement `solve(P, Q, beta=0.5)`, returning a 0-dimensional scalar tensor. `P` and `Q` are float32, shape \((N, C)\), **row-normalized** probability distributions. Following the formula above: first compute the per-row mixture distribution \(M=\beta P+(1-\beta)Q\), then compute \(\beta\,\mathrm{KL}(P\|M)+(1-\beta)\,\mathrm{KL}(Q\|M)\), where \(\mathrm{KL}(A\|B)=\sum_c A_c\log(A_c/B_c)\) with the convention \(0\log 0=0\); after obtaining one JSD value per row, take the **mean** over the \(N\) rows.

## Examples

- \(N=1\), `P=[[1.0,0.0]]`, `Q=[[0.0,1.0]]`, `beta=0.5`: the two distributions have disjoint support, \(M=[[0.5,0.5]]\), \(\mathrm{JSD}=\ln 2\approx 0.69314718\) (attains the upper bound).
- `P=[[0.7,0.3],[0.2,0.8]]`, `Q=[[0.4,0.6],[0.5,0.5]]`, `beta=0.5`: \(\mathrm{JSD}=0.04843632\).

## Constraints

- CPU float32; each row of `P` and `Q` is a normalized probability distribution.
- \(0\log 0 = 0\); since \(M\) lies between the two distributions, \(P_c>0\Rightarrow M_c>0\), so no \(\log\) division by zero occurs.
- For `beta=0.5` the result is symmetric under swapping `P` and `Q`; for any `beta`, \(0\le \mathrm{JSD}\le \ln 2\) (natural logarithm).
- When `P` and `Q` are identical, the return value should be approximately 0.
