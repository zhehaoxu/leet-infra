# PowerNorm and Additive Attention Pooling Readout

## Problem

Normalization and pooling are the two standard readout utilities in front of a sequence model's head. PowerNorm (Shen et al., *PowerNorm: Rethinking Batch Normalization in Transformers*, ICML 2020, arXiv:2003.07845) cuts BatchNorm/LayerNorm's "subtract mean, divide by std" down to dividing by a single **second-order power mean**: per spatial location, the denominator uses only the mean of squared channels \(S=\frac1C\sum_c x_c^2\), giving \(y=x/\sqrt{S+\varepsilon}\) — no mean subtraction, no variance estimate, and therefore **never centering** on inputs with a nonzero mean. On the pooling side, Set Transformer's PMA readout (Lee et al., *Set Transformer*, arXiv:1810.00825) attends over set elements with a seed vector; this problem implements its single-query degenerate form:

\[\mathrm{out}=\mathrm{softmax}(xq/\sqrt D)^\top x.\]

Two modes, mutually exclusive: `"powernorm"` and `"attn-pool"`, and **each mode ignores the other's parameters**. CPU, fully deterministic, no RNG calls.

## Rules

- The full signature, parameter meanings and defaults live only in the starter docstring; it is the single judging standard. Two modes: `"powernorm"` / `"attn-pool"`, any other value raises; the input must be a float32/float64 tensor (other dtypes raise); each mode has its own dimensionality requirement, and violations raise.
- **powernorm** (pure second-moment form): input is (N,C,L) 3-D or (N,C,H,W) 4-D; flatten to (N,C,M) (4-D: \(M=H\cdot W\), spatial dims collapse entirely); per (n,m) location compute the channel mean square \(S(n,m)=\frac1C\sum_c x[n,c,m]^2\), then

  \[y[n,c,m]=x[n,c,m]\,/\,\sqrt{S(n,m)+\varepsilon}.\]

  **No centering of any kind** (a dedicated probe checks this); \(\varepsilon>0\) enters the denominator only, guarding zero inputs (\(0/\sqrt\varepsilon=0\)). Returns the same shape and dtype as the input.
- **attn-pool** (single-query attention pooling, no temperature): input is (B,L,D) 3-D; the query vector must have length D and may carry a different dtype (cast to the input dtype before the computation); scoring

  \[\text{logits}[b,l]=\Big(\sum_d x[b,l,d]\,q_d\Big)\big/\sqrt D,\]

  a numerically stable softmax (subtract max) over the L dimension (dim=-1) gives weights of shape (B,L); \(\text{out}[b,d]=\sum_l\text{attn}[b,l]\,x[b,l,d]\), shape (B,D), dtype same as the input.
- **The two modes ignore each other's parameters**: in powernorm mode the query vector takes no part at all (even an invalid value must not affect the result), and in attn-pool mode eps takes no part; a missing query vector, a non-1-D query, or a query whose length ≠ D raises; eps must be a positive number, otherwise raise (checked in powernorm mode only).
- The softmax must be numerically stable (subtract max, then exp-normalize). Mathematical properties: all logits equal → uniform weights \(1/L\) and out == the elementwise mean; L=1 → out **bitwise equal** to that row.

## Example

Input of shape (1,2,4) (\(N{=}1,C{=}2,L{=}4\)), default eps \(10^{-6}\):

\[x=\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix}\]

**powernorm**: per column \(S=(x_0^2+x_1^2)/2=[13,\,20,\,29,\,40]\), denominator \(\sqrt{S+10^{-6}}\approx[3.60555,\,4.47214,\,5.38516,\,6.32456]\), output (the EXAMPLE_PN backfill in test_cases.py, 10 decimal places):

\[y=\begin{bmatrix}0.2773500681&0.4472135901&0.5570859909&0.6324555278\\1.3867504597&1.3416407108&1.2998672724&1.2649110556\end{bmatrix}\]

**attn-pool**: the same matrix viewed as (B=1,L=2,D=4) with query \(q=[1,-1,0.5,0]\): logits \(=[0.25,\,1.25]\); after softmax the weights are \([1/(1{+}e),\,e/(1{+}e)]=[0.2689414214,\,0.7310585786]\), output (EXAMPLE_ATTN):

\[\mathrm{out}=[3.9242343903,\;4.9242343903,\;5.9242343903,\;6.9242343903]\]

Contrast: the elementwise mean is \([3,4,5,6]\) — out collapses to the mean only when the scores are all equal; here the logits differ by 1.0 and the weights tilt toward the second row.

## Constraints

- mode accepts only `"powernorm"`/`"attn-pool"`; eps is used only in powernorm mode and must be positive; the attn-pool query must be 1-D with length equal to D and may differ in dtype; wrong dimensionality (powernorm not 3-D/4-D, attn-pool not 3-D) and illegal dtypes all raise ValueError
- Outputs: powernorm same shape and dtype as input; attn-pool (B,D) with the input's dtype; the float64 oracle case is compared at 1e-12, float32 cases at 1e-6
- Discriminating probes asserted directly: on inputs with a nonzero channel mean, powernorm does **not** center (significantly different from the BatchNorm-style \((x-\mu)/\sqrt{\mathrm{var}+\varepsilon}\), and the output's channel mean is clearly nonzero); with tiny eps the per-location channel RMS ≈ 1, and a larger eps deviates as \(\sqrt{S/(S+\varepsilon)}\) (matched pointwise against theory); constant input → \(\mathrm{sign}(c)\), zero input → 0, elementwise sign preserved; equal query dot products → out == mean; L=1 → torch.equal with that row
