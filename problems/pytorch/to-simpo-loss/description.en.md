# Alignment Training: SimPO Length-Normalized Preference Loss

## Problem

**SimPO** (Simple Preference Optimization, Meng et al., arXiv:2405.14734) uses the **length-normalized** average log-probability as the "reward", drops DPO's reference model, and adds a **target margin** \(\gamma\) to widen the gap between the better and worse samples:

\[ r_\theta(x,y) = \frac{1}{|y|}\log \pi_\theta(y|x) \]

The loss (a sigmoid-style objective over the preference pair \(y_w \succ y_l\)):

\[ \mathcal{L} = -\log\sigma\bigl(\beta\,(r_\theta(x,y_w) - r_\theta(x,y_l)) - \gamma\bigr) \]

Here the given `logp_w` and `logp_l` are the sequence-level log-probabilities \(\log\pi_\theta(y|x)\); dividing by the response length gives \(r\).

## Rules

Inputs are `logp_w`, `logp_l` (sequence log-probabilities), `len_w`, `len_l` (response token counts), all of shape \((N,)\), plus the scalars `beta` (temperature) and `gamma` (target margin). Per sample:
\[ r_w=\frac{\text{logp}_w}{\text{len}_w},\quad r_l=\frac{\text{logp}_l}{\text{len}_l},\quad \mathcal{L}=-\frac1N\sum_i\log\sigma\bigl(\beta(r_{w,i}-r_{l,i})-\gamma\bigr) \]
Return a 0-dim float32 scalar. Do not use an off-the-shelf alignment library.

## Example

`logp_w=[-2.0,-1.5]`, `len_w=[4,3]`, `logp_l=[-3.0,-2.5]`, `len_l=[5,4]`, `beta=2.0`, `gamma=0.5`:

```
r_w = [-0.5, -0.5] ;  r_l = [-0.6, -0.625]
margin = 2*(r_w-r_l) - 0.5 = [-0.3, -0.25]
loss = -mean(logsigmoid(-0.3), logsigmoid(-0.25)) = 0.840147
```

## Constraints

- CPU float32; `len_w`, `len_l` are positive; \(\beta>0\).
- Length normalization is the key difference between SimPO and DPO (it removes the bias where "long responses accumulate more log-probability").
- With \(\gamma=0\), unnormalized rewards and a reference model, it approaches DPO; this problem implements only the formula above.
