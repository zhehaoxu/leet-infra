# In Depth: PowerNorm and Additive Attention Pooling Readout

## Source

- PowerNorm: Shen et al., *PowerNorm: Rethinking Batch Normalization in Transformers* (ICML 2020, arXiv:2003.07845). The paper's systematic study attributes BN's failure in Transformers to the violent fluctuation of the variance estimate under small batches/long sequences, and estimates central moments via Taylor expansion; this problem takes only the most basic **non-centralized second-moment form** (plain \(x/\sqrt{\mathrm{mean}(x^2)}\)), exactly as the starter docstring specifies — no Taylor-centralized variant involved.
- Additive attention readout: Lee et al., *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks* (arXiv:1810.00825), whose PMA (Pooling by Multi-head Attention) produces k readouts from k seed vectors; this problem is the **single-seed degeneration**: single head, single query, dot-product score scaled by \(1/\sqrt D\), softmax-weighted sum of values.
- The RMS≈1 property, the non-centering probe and the constant-input behavior are defined solely by the starter docstring / test_cases.py discriminating cases.

## Full derivation and index convention

**Index convention**: powernorm's input flattens to (N,C,M); at location \((n,m)\), \(S(n,m)=\frac1C\sum_c x[n,c,m]^2\) and \(y[n,c,m]=x[n,c,m]/\sqrt{S(n,m)+\varepsilon}\); for 4-D (N,C,H,W) input, \(M=H\cdot W\), spatial dims collapse entirely and **the channel dim is the only statistics axis**. attn-pool's input is (B,L,D); weights normalize along L (dim=-1); output (B,D).

**Why "no mean subtraction" is discriminable**: BatchNorm/LayerNorm's denominator is the centralized variance \(\mathrm{var}=\overline{x^2}-\bar x^2\). Whenever the channel mean \(\bar x\neq0\), \(\sqrt{\mathrm{var}+\varepsilon}\) diverges from this problem's \(\sqrt{\overline{x^2}+\varepsilon}\). Extreme contrast: all channels equal to the same constant \(c\) — BN's denominator → \(\sqrt\varepsilon\) (zero variance, output blows up to \(\pm c/\sqrt\varepsilon\)), while here \(S=c^2\) and \(y=c/|c|=\mathrm{sign}(c)=\pm1\) exactly. The probe watches two observables: ① the output is not allclose to an independent BatchNorm-style recompute; ② the output's channel mean is clearly nonzero — a centered normalizer's output channel mean is **always 0**, a non-centered one's **cannot be 0**, the most direct physical evidence of "did we subtract the mean".

**Exact form of RMS≈1**: feed the output back into the definition, \(\mathrm{mean}_c(y^2)=\dfrac{S}{S+\varepsilon}\), so each (n,m) location's channel RMS \(=\sqrt{S/(S+\varepsilon)}\). With \(\varepsilon\ll S\) the RMS ≈ 1 (normalization active); with eps comparable to S it visibly drops below 1 — the probe fixes the input at \(S\approx4.67/1.17\), applies \(\varepsilon=0.25\), and matches the theoretical \(\sqrt{S/(S+\varepsilon)}\approx0.974/0.907\) pointwise. PowerNorm's "unit scale" is thus an **approximate property controlled by the relative size of eps**, unlike the strictly zero-mean/unit-variance output of centered normalization.

**Constant and zero inputs**: \(x\equiv c\) (with \(|c|\gg\sqrt\varepsilon\)) → \(y\equiv c/|c|=\mathrm{sign}(c)\); \(x\equiv0\) → \(S=0\), \(y=0/\sqrt\varepsilon=0\). The divisor is always positive, so elementwise signs are preserved (the sign probe). All of these are immediate corollaries of "divide by a positive number", and also evidence that eps belongs in the denominator only — anywhere else and the zero input would stop mapping to 0.

**Two identities of attn-pool**:

1. **Uniform weights**: softmax is shift-invariant — when all logits are equal, every weight is exactly \(1/L\); the common score cancels inside the max subtraction, and \(\mathrm{out}=\frac1L\sum_l x_l=\mathrm{mean}\). The probe builds \(q=e_0\) with all \(x_{l,0}=1\) (dot product identically 1) and asserts out == the elementwise mean. This also shows scoring acts only through **relative differences** into the softmax: a global shift of the logits changes nothing.
2. **L=1**: a single-element softmax is exactly 1 (subtract max → exp(0)/exp(0)=1), so out is **bitwise** equal to that row (torch.equal, not allclose) — neither exp nor division introduces any numeric error.

**Why the score divides by \(\sqrt D\)**: logits \(=xq/\sqrt D\) shares the standard dot-product-attention scaling — with q fixed, \(xq\) sums D products and grows like \(\sqrt D\); without the scale the softmax saturates toward a one-hot. There is no temperature parameter here; \(1/\sqrt D\) is the only scaling (the temperature-softmax primitive lives in a neighboring problem).

## Key points and common mistakes

1. **Do not center by reflex**: LN/BN muscle memory writes the variance \(\overline{x^2}-\bar x^2\) instead of \(S\) — the non-centering probe catches exactly this (nonzero output channel mean is the hard indicator).
2. **eps only in the denominator**: not added to the input, not applied after the division; it is the zero-input backstop (0, not NaN).
3. **4-D flattening**: H·W collapses into M; statistics run along the channel dim C only — folding spatial positions into the statistics axis is wrong.
4. **Softmax over dim=-1 (the L dim)**: weights normalize over the sequence dim, not D; align (B,1,L)×(B,L,D) before the weighted sum.
5. **Subtract the max in softmax**: mathematically equivalent without, but large logits overflow; the deterministic contract requires the stable form.
6. **Query dtype cast**: the query may carry a different dtype; cast to the input's dtype before the dot product; the returned dtype is always the input's.
7. **"Ignore" is not "validate"**: powernorm must ignore the query even if it is malformed, and attn-pool must ignore eps — implementing "ignore" as "validate" kills legal calls.
8. **Error surface**: illegal mode, non-float dtype, wrong dimensionality, missing/non-1-D/wrong-length query, eps≤0 — all ValueError.

## Differences from neighboring problems

- **vs `batchnorm-forward` / `layernorm-forward` / `rmsnorm`**: the centered-normalization family. BN subtracts batch statistics, LN subtracts over the feature dim; RMSNorm shares this problem's "divide by RMS only" but normalizes along the feature/channel dim with a learnable scale, while this problem computes statistics per spatial location (n,m) over channels, has no scale parameter, and eps only guards zero inputs.
- **vs `global-pooling`**: global-pooling is a pure statistic readout (mean/max, no weights); this problem's pooling weights are data-dependent (softmax scores). The uniform-weights probe gives the degeneration: equal scores collapse attn-pool into global average pooling.
- **vs `attention-softmax-temp`**: that problem is a temperature-softmax primitive over decoding logits with an explicit, tunable T; here the softmax acts on \(xq/\sqrt D\) scores with no temperature.
- **vs `multi-head-attention`**: MHA is token-to-token self-attention (Q comes from the input, output stays sequence-shaped); this problem reads a set down to a single point — the query is an external seed and the output has length 1.

## Cross-check convention and complexity

- Independent recompute `_oracle_pn`: pure Python, loops over (n,m) locations, explicitly accumulates squared channels and divides, all in float64 then cast back — a different path from the vectorized mean/pow solution; a separately implemented eps-parameterized oracle avoids a shared-bug surface. attn-pool's `_oracle_attn` hand-writes the subtract-max softmax plus the explicit weighted sum, covering three float32 cases and one float64 case (1e-12).
- Discriminating probes: non-centering (significantly different from the BatchNorm-style recompute, nonzero output channel mean), RMS \(=\sqrt{S/(S+\varepsilon)}\) matched to theory, constant → sign(c), zero → 0, sign preservation, uniform weights → mean, L=1 bitwise equality, plus 10 ValueError error-surface cases.
- Complexity: powernorm \(O(N\cdot C\cdot M)\) time (one pass for the squared sum, one for the division), intermediates of the same order; attn-pool \(O(B\cdot L\cdot D)\) time (score dot product + softmax + weighted sum, one pass each), \(O(B\cdot L)\) extra space for the weights.
