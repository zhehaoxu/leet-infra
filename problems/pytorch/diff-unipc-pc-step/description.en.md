# Diffusion sampling: the UniPC predictor-corrector step reads two different histories

## Description

`UniPCMultistepScheduler.step()` does two things in one call: it first uses the **current** network output to look **backwards** and correct the previous prediction, then rotates the buffer and steps **forwards**. This problem asks for exactly that pair of updates. The inputs are already in the scheduler's internal convention (an \(x_0\) prediction when `predict_x0=True`, an ε prediction otherwise); no network is involved.

All the difficulty sits in two asymmetries. First, **the two sides do not share their nearest history**: upstream rotates the buffer only *after* the corrector, so the corrector's \(m_0\) is the previous step's output with older entries starting at `m_hist[-2]`, while the predictor's \(m_0\) is the **current** output `m_t` with older entries starting at `m_hist[-1]`; the \(\lambda\) indices each side reads are one slot apart as well. Second, **the orders that get pinned to 0.5 differ**: the predictor skips the linear solve at `order_p == 2`, the corrector skips it at `order_c == 1`. And since \(b_i\propto 1/B_h\), the product \(B_h\cdot\text{rhos}\) is independent of `solver_type` whenever the general solve is used - so `bh1` versus `bh2` moves only those two pinned branches, and every other configuration is bitwise identical.

## Rules

All intermediates are computed in float64 and the two returned tensors are cast to float32. Write `s = sigmas`.

**Schedule**: with `use_flow_sigmas=False` (VP), \(\alpha_i=1/\sqrt{s_i^2+1},\ \tilde\sigma_i=s_i\alpha_i\); with it true (flow), \(\alpha_i=1-s_i,\ \tilde\sigma_i=s_i\). Then \(\lambda_i=\log\alpha_i-\log\tilde\sigma_i\).

**Shared machinery for one side** (\(r\) is that side's order, \(s_0\) the base index, \(t\) the target index, \(h=\lambda_t-\lambda_{s_0}\)):

\[ hh=\begin{cases}-h & \text{predict\\_x0}\\[2pt] h & \text{otherwise}\end{cases}\qquad \varphi_1=\operatorname{expm1}(hh)\qquad B_h=\begin{cases}hh & \text{bh1}\\ \operatorname{expm1}(hh) & \text{bh2}\end{cases} \]

For \(i=1\dots r-1\) take that side's history entry \(m_i\) and index \(s_i\), set \(rk_i=(\lambda_{s_i}-\lambda_{s_0})/h\) and \(D_1^{(i)}=(m_i-m_0)/rk_i\); append \(rk_r=1\) at the end, and row \(i\) of \(R\) is \((rk_1^{\,i-1},\dots,rk_r^{\,i-1})\). The scalar sequence is

\[ v_1=\frac{\varphi_1}{hh}-1,\quad f_1=1,\qquad b_i=\frac{v_i f_i}{B_h},\quad f_{i+1}=(i+1)f_i,\quad v_{i+1}=\frac{v_i}{hh}-\frac{1}{f_{i+1}} \]

note that upstream updates \(f\) **before** producing the next \(v\); writing \(v_{i+1}=v_i/hh-1/f_i\) is wrong.

**The coefficients \(\text{rhos}\)**: the corrector uses `[0.5]` when `order_c == 1` and \(\operatorname{solve}(R,b)\) otherwise; the predictor uses `[0.5]` when `order_p == 2`, an empty list when `order_p == 1`, and otherwise \(\operatorname{solve}(R_{<r,\,<r},\,b_{<r})\), i.e. the last row **and** the last column are both dropped.

**Corrector side** (skipped entirely when `x_last` is None): \(s_0=\)`step_index` \( -1\), \(t=\)`step_index`; \(m_0=\)`m_hist[-1]`, \(m_i=\)`m_hist[-(i+1)]`, \(s_i=\)`step_index` \(-(i+1)\). With \(m_t=\)`m_t` supplying the last term,

\[ c=\sum_{i<r}\text{rhos}_i D_1^{(i)}+\text{rhos}_r\,(m_t-m_0) \]

**Predictor side**: \(s_0=\)`step_index`, \(t=\)`step_index` \(+1\); \(m_0=m_t\), \(m_i=\)`m_hist[-i]`, \(s_i=\)`step_index` \( - i\), and \(p=\sum_{i}\text{rhos}_i D_1^{(i)}\) (as many terms as there are \(D_1\)s).

**Update** (identical shape on both sides; \(x_{s_0}\) is `x_last` on the corrector and the corrected value - or `x_this` - on the predictor):

\[ x_t=\begin{cases}\dfrac{\tilde\sigma_t}{\tilde\sigma_{s_0}}\,x_{s_0}-\alpha_t\varphi_1 m_0-\alpha_t B_h(\text{combined differences}) & \text{predict\\_x0}\\[6pt] \dfrac{\alpha_t}{\alpha_{s_0}}\,x_{s_0}-\tilde\sigma_t\varphi_1 m_0-\tilde\sigma_t B_h(\text{combined differences}) & \text{otherwise}\end{cases} \]

Return the pair `(corrected, prev)`; when `x_last` is None, `corrected` is `x_this` passed through unchanged.

## Examples

With `sigmas = [8, 4, 2, 1, 0.5, 0.25, 0.125]` (VP) the schedule is perfectly symmetric around \(\lambda=0\), and at `step_index=3` both sides have step \(h=0.693147\), \(hh=-0.693147\):

```
i     sigma      alpha   sigma~      lambda
0   8.000000  0.124035  0.992278   -2.079442
1   4.000000  0.242536  0.970143   -1.386294
2   2.000000  0.447214  0.894427   -0.693147
3   1.000000  0.707107  0.707107    0.000000
4   0.500000  0.894427  0.447214    0.693147
5   0.250000  0.970143  0.242536    1.386294
6   0.125000  0.992278  0.124035    2.079442
```

Intermediates for `order_c=2, order_p=2, solver_type="bh2"` (\(rk_r=1\) is listed last):

```
corrector  rks=[-1.000000, 1.000000]  phi1=-0.500000  B_h=-0.500000
           R=[[1.000000, 1.000000], [-1.000000, 1.000000]]
           b=[0.557305, 0.391958]     rhos=[0.082674, 0.474631]
predictor  rks=[-1.000000, 1.000000]  phi1=-0.500000  B_h=-0.500000
           b=[0.557305, 0.391958]     rhos=[0.5]        (pinned, no solve)
```

Given

```
m_hist[0] = [[ 0.4, -0.2], [ 1.1, 0.3]]
m_hist[1] = [[-0.5,  0.7], [ 0.2, 0.9]]
m_hist[2] = [[ 0.6,  0.1], [-0.3, 0.5]]
m_t       = [[ 0.2, -0.4], [ 0.8, 0.15]]
x_last    = [[ 1.3, -0.9], [ 0.4, 1.7]]
x_this    = [[ 1.1, -0.7], [ 0.6, 1.5]]
```

with `step_index=3`, `predict_x0=True` and defaults elsewhere:

```
(2,2) bh2     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.762047, -0.782485], [0.844157, 0.906083]]
(2,2) bh1     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.727496, -0.825674], [0.939173, 0.875850]]
(2,3) bh2     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.433403, -0.774073], [1.211966, 0.907726]]
(2,3) bh1     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.433403, -0.774073], [1.211966, 0.907726]]
(1,1) bh2     corrected=[[1.169162, -0.764545], [0.404616, 1.458873]]  prev=[[0.828885, -0.662426], [0.613673, 0.989754]]
(1,1) bh1     corrected=[[1.141846, -0.798689], [0.479733, 1.434972]]  prev=[[0.811610, -0.684021], [0.661180, 0.974638]]
```

In the two `(2,2)` rows `corrected` matches while `prev` differs by 0.095016; in the two `(1,1)` rows `corrected` itself already differs by 0.075117. The first is the predictor's pinned 0.5, the second the corrector's. The two `(2,3)` rows are identical because both sides take the general solve. With `predict_x0=False`, `(2,3)` gives

```
corrected=[[1.668741, -1.331373], [0.593786, 2.446867]]  prev=[[2.342067, -1.422294], [-0.082072, 3.089230]]
```

With `x_last=None` the corrected value is returned unchanged and `prev` extrapolates directly from `x_this`.

## Constraints

- `m_t / m_hist[*] / x_this` (and `x_last` when not None) must share one shape, be at least 2-D (dimension 0 is the batch), and contain only finite values; `m_hist` is a non-empty sequence.
- `sigmas` is a 1-D, finite, **strictly positive** tensor (the terminal \(\sigma=0\) step is out of scope here: with `bh1` it yields NaN or ±inf) and `len(sigmas) >= step_index + 2`.
- `step_index / order_c / order_p` are Python ints (`bool` is illegal) with `order_c, order_p >= 1`; enabling the corrector requires `step_index >= order_c` and `len(m_hist) >= order_c`, and in every case `step_index >= order_p - 1` and `len(m_hist) >= order_p - 1`.
- `use_flow_sigmas / predict_x0` are bools; under the flow convention \(\sigma\ge1\) destroys the finiteness of \(\lambda\) and raises `ValueError`.
- `solver_type` accepts `bh1` / `bh2`; upstream silently maps `midpoint` / `heun` / `logrho` to `bh2` and this problem keeps that mapping. Any other name raises `ValueError`.
- `import diffusers` is forbidden.
