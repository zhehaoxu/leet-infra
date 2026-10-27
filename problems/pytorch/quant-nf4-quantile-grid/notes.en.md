# Quantization: NF4 Quantile Codebook Construction and Block Encoding — Algorithm Notes

## 1. Why quantiles instead of a uniform grid

4-bit uniform quantization cuts \([-1,1]\) into 16 equal-width bins, but normal weights are **dense in the middle and sparse in the tails**: equal-width grids waste resolution at 0 and in the tails. NF4 switches to **equal-probability** bins — each bin carries \(1/16\) of the weight mass, so bins are naturally narrower near 0 and wider in the tails, minimizing per-value MSE (the ablation in Dettmers et al., arXiv:2305.14314 shows NF4's reconstruction error well below int4/fp4).

## 2. The asymmetric bitsandbytes construction (judged point)

What `create_normal_map(offset=0.9677083, use_extra_value=True)` actually does (this problem reproduces it to \(10^{-7}\)):

- Positive half, **8** probability points: `linspace(offset, 0.5, 9)[:-1]`, step \((0.5-\text{offset})/8\);
- Negative half, **7** probability points: `linspace(offset, 0.5, 8)[:-1]`, step \((0.5-\text{offset})/7\);
- Add one exact \(0\) (the symmetric midpoint of the normal), 16 values in total;
- Sort and divide by `max|v|` to normalize into \([-1,1]\).

`offset` is the probability of the largest quantile: `ppf(0.9677083)≈1.845`, i.e. mass beyond \(\pm1.845\sigma\) saturates to \(\pm1\). The positive side has one more point than the negative side, so the codebook is **not symmetric** about 0 — `check_codebook_anchors` pins this down with "the first positive gap is smaller than the first negative gap" (a naive symmetric `linspace(0.5,1,9)` fails immediately).

The exact table (float64) lives in the `EXACT` constant of `test_cases.py`: `-0.6961928 / 0.0795803 / 0.7229568 / 1.0`, etc. **Note**: the problem statement of `peft-qlora-nf4-forward` gives the paper's Appendix E table rounded to four decimals (`0.7226`), which differs from the exact value by \(3.6\times10^{-4}\) — each problem is self-consistent and neither references the other.

## 3. Block-scaled encoding (judged point)

QLoRA's 4-bit storage is "16-value lookup + one absmax per 64 weights". This problem covers only the **encoding** side (dequantized forward lives in `peft-qlora-nf4-forward`):

\[
s=\max_{j\in\text{block}}|W_{ij}|,\qquad u=W_{ij}/s,\qquad c=\arg\min_k|u-T_k|,\qquad \hat W=T_c\,s
\]

- Ties go to the smaller code: `argmin`'s default, stated explicitly to avoid implementation drift;
- Zero block absmax (all-zero block): code fixed at 7, reconstruction 0 — `check_matches_oracle` asserts it with an all-zero row;
- **Scaling linearity**: multiplying a block by \(\lambda\) leaves \(u\) and the code unchanged while \(\hat W\) scales exactly by \(\lambda\) (multiplication by 2 is exact in floating point) — `check_block_scaling_linearity` uses \(\lambda=2\). This is the structural invariant that "codebook and scale are decoupled"; any implementation that mixes scale into the codebook (or vice versa) breaks it.

## 4. Implementation notes

- Use `torch.erfinv` for `Φ⁻¹`: `Φ⁻¹(p)=√2·erfinv(2p−1)`; vectorize in float64.
- Encode by broadcasting: `(blocks/safe).unsqueeze(-1) - table` gives `(M, N/64, 64, 16)`, then `argmin(-1)`; memory is \(16\times\), harmless at judging sizes (≤ 4×128).
- Override degenerate blocks with `torch.where` so `0/0` cannot inject NaN into `argmin`.

## 5. Division of labor within the collection

- `symmetric-quantize` / `per-channel-quantize` / `group-wise-quantize`: linear uniform grids + absmax scaling;
- `gptq-column-quantize`: error-compensated column-wise quantization;
- `peft-qlora-nf4-forward`: dequantized forward + double-quant byte accounting **given** NF4 codes and absmax;
- This problem: **how the codebook is built** (quantiles + normalization) and **how to encode** (nearest lookup). Scale-level second quantization is `quant-double-quant-scale`.

## 6. Verification

- `check_example`: \(1\times64\), absmax=1, hand-computed codes `[15,0,15,12]` and reconstruction.
- `check_codebook_matches_reference`: value-by-value against the exact bitsandbytes table (tolerance \(10^{-5}\)).
- `check_codebook_anchors`: monotone, \(\pm1\), code 7 is zero, asymmetric structure.
- `check_matches_oracle`: random \((4,128)\) plus an all-zero row, scalar-by-scalar recomputation (`equal` for codes, \(10^{-5}\) for reconstruction).
- `check_block_scaling_linearity`: scaling linearity + `offset` sensitivity + invalid `offset` raising `ValueError`.
