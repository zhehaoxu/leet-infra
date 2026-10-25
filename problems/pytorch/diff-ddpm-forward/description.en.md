# Diffusion: DDPM Forward Noising q(x_t|x_0)

## Problem

The DDPM forward process noises a clean sample \(x_0\) to step \(t\) directly "in one shot": there is no need to actually iterate \(t\) times, because Gaussians can be merged and the result is still a normal distribution with known mean and variance —

\[ x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I) \]

where \(\bar\alpha_t\in(0,1]\) is the signal retention ratio (given by the noise schedule). The noise \(\epsilon\) is passed in explicitly as an argument to make grading deterministic.

## Rules

Given `x0`, `eps` of the same shape, and a scalar `alpha_bar` (\(\bar\alpha_t\)), return

\[ x_t = \sqrt{\bar\alpha_t}\cdot x_0 + \sqrt{1-\bar\alpha_t}\cdot \epsilon \]

with the same shape as `x0`, float32. Ready-made diffusion libraries are forbidden.

## Examples

`alpha_bar=0.6`, \(x_0=\begin{bmatrix}1.0&-2.0\\ 0.5&3.0\end{bmatrix}\), \(\epsilon=\begin{bmatrix}0.3&-0.7\\ 1.2&-0.4\end{bmatrix}\):

```
x_t = [[0.9643333554267883, -1.9919122457504272],
       [1.146245002746582 ,  2.070807933807373 ]]
```

## Constraints

- `alpha_bar` is a float scalar in \((0,1]\); `x0` and `eps` have the same shape.
- When \(\bar\alpha_t=1\), \(x_t=x_0\) (no noise); as \(\bar\alpha_t\to0\), it approaches the pure-noise direction \(\epsilon\).
- float32 throughout, element-wise operations, supporting any shape (including high-dimensional).
