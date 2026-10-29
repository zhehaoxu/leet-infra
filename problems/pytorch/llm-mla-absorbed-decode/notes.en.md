# Deep Dive: Weight Absorption for MLA Decode

## Where it comes from

Multi-head Latent Attention from DeepSeek-V2 (arXiv:2405.04434; §3.1 defines the MLA structure and \(W_{UK},W_{UV}\)). The absorption identity is just associativity of two linear maps and follows directly from the definitions:

\[ q_h^\top\left(W_{UK,h}c_{KV}\right)=\left(q_h^\top W_{UK,h}\right)c_{KV},\qquad \sum_t p_t\left(W_{UV,h}c_{KV,t}\right)=W_{UV,h}\left(\sum_t p_t c_{KV,t}\right) \]

This problem covers only the **decode** side: queries and positional encoding are already computed, and the inputs are the per-head \(q^{\text{nope}},q^{PE}\), the latent cache, and the two families of lifting matrices — no \(W_{DQ}/W_{UQ}/W_{QR}\) projection chain. The repo's `llm-deepseek-mla` problem covers the unabsobed full-layer forward (q-LoRA path, RMSNorm, decoupled RoPE, causal attention, `o_proj`), so the two form a pair: **the same mathematical object under two computation graphs**, which makes the non-absorbed forward a natural independent oracle.

## Key ideas and pitfalls

1. **The coordinate system of \(W_{UK},W_{UV}\)**: both are \((H,d_h,d_c)\), "row \(k\) holds the coefficients of the \(k\)-th coordinate of \(k^{\text{nope}}\) over the latent". Absorbing into the query contracts over rows (\(q_h^\top W_{UK,h}\), yielding \(d_c\)); lifting the output expands over columns (\(W_{UV,h}z_h\), yielding \(d_h\)). Swapping these two einsum subscripts — and getting away with it whenever \(d_h=d_c\) — is the classic mistake; the judge deliberately uses random cases with \(d_h\ne d_c\).
2. **The RoPE part is never absorbed**: \(k^{PE}\) is shared across heads and unrelated to the latent cache, so it can only enter as a separate \(q^{PE}\cdot k^{PE}\) term added to the scores. Folding \(q^{PE}\) into \(W_{UK}\) breaks the dimensions outright; treating \(k^{PE}\) as per-head introduces differences between heads that must not exist.
3. **The value sum can be reordered**: \(\sum_t p_t (W_{UV}c_t)=W_{UV}(\sum_t p_t c_t)\). The lift then happens once instead of once per token — that reordering is exactly what saves decode memory and bandwidth. Lifting per token and summing afterwards is equivalent but discards the whole point.
4. **The score scale is applied once**: `scale` multiplies the sum of the latent inner product and the RoPE inner product, not each term separately by different factors. (A single scalar multiplying both is algebraically the same, but forgetting to scale the RoPE term changes the attention distribution.)
5. **Softmax runs along the sequence axis**: `dim=-1` is \(S\), neither the head axis nor the feature axis.
6. **Degenerate checks are powerful**: with \(S=1\) the weight is identically 1, so the output is independent of the query and equals \(W_{UV,h}c_{KV,0}\); with zero queries and zero RoPE keys the weights are uniform and the output is \(W_{UV,h}\cdot\text{mean}_t(c_{KV})\). Both need no randomness and immediately expose "the query/position term leaked into the value path".

## How the judge cross-checks

The reference solution runs the absorbed path in float64 with three einsums (fold the query, score, latent-space weighted sum plus one lift) and casts to float32. The `_oracle` takes the **naive path**: it first lifts each token's \(k^{\text{nope}}\) and \(v\) per head with `einsum("bsc,hkc->bhsk")`, then scores ordinarily and sums per token — the two graphs share no intermediate quantity. Six structural assertion groups: three \((B,H,d_h,d_r,d_c,S)\) configurations (including \(d_h\ne d_c\)) at two scales, the query-independence and exact value at \(S=1\), uniform attention for zero queries, head independence, the RoPE key's separate contribution, and invalid inputs. Tolerance `allclose(atol=1e-5)`.

## Complexity

With sequence length \(S\), \(H\) heads, latent width \(d_c\), and head width \(d_h\): the absorbed path costs \(O(H d_h d_c)\) to fold the query, \(O(BHSd_c)\) for scores, \(O(BHSd_c)\) for the latent weighted sum, and \(O(BH d_h d_c)\) for the lift. Everything scaling with \(S\) acts on \(d_c\) only, and no \(S\times H\times d_h\) intermediate is ever materialized; the naive path spends \(O(BHSd_h)\) on attention and materializes an intermediate of that size. The bandwidth advantage is largest when \(S\) is small and \(d_c\ll H d_h\), i.e. exactly at decode time.
