# Solution Notes · SpQR Sparse Outliers

## Demoting "outliers" to isolated points

The granularity of LLM.int8() is the **column**: a single large value anywhere in a column sends the whole column back to FP16. But in real weights a "bad column" usually has only one or two genuinely absurd elements, and dragging the whole column along inflates the average bit width noticeably. SpQR shrinks the definition of an outlier down to the **individual element** — $|W_{oj}|>\tau$ — extracts them as a **sparse matrix** stored in FP16, and the dense bulk can then be safely squeezed to very low bits. Since the sparse part is stored in a format such as CSR and only added back at multiply time via gather, the bit-width cost depends only on the **number** of outliers, not on how many columns they happen to sit in.

## Per-row quantization

Another detail of this problem: the range of the dense bulk is not a global per-tensor one but **per row**, $s_o=\max_{j\in\text{non-out}}|W_{oj}|$. Weight magnitude distributions differ a lot across output channels, and sharing one global scale would squash a "narrow" row entirely into the lowest few grid points. Taking each row's own absmax markedly improves the use of the effective bit width. Outlier elements take **no part** in their own row's scale (otherwise a single large value would blow the range up again, as if nothing had been exempted).

## The granularity spectrum of the three siblings

| Method | Outlier test | Granularity | Protection |
|---|---|---|---|
| LLM.int8() | column absmax > τ | whole column | keep the column in FP |
| **SpQR** | element-wise \|W\|>τ | single point | sparse FP + per-row quantization |
| AWQ | activation importance | whole column | scale first, quantize together |
| QuaRot | — (eliminate outliers) | global | orthogonal rotation spreads them out |

SpQR's core advance over LLM.int8() is precisely **refining the exemption granularity from columns to elements**, keeping as few high-precision protected weights as possible.

## Edge-case self-checks

- `check_finer_than_column`: within one column, `[9, 0.1]` — only `9` is exempted while `0.1` is quantized normally by its row's scale, directly verifying "element-wise" rather than "column-wise".
- `check_all_outlier_passthrough`: an extremely low threshold makes everything an outlier → output equals input, no quantization.
- `check_matches_oracle`: 30 random sets of $W$/$\tau$/$bits$, cross-checked against an independent pure-Python per-row implementation (mask with `torch.equal`, reconstruction with `allclose`).
