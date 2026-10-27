# In Depth: 4-D Additive Attention Mask Construction

## Source

- Additive mask and the finfo.min convention: HuggingFace transformers' masking utilities and flash-attn's add-to-logits mask convention — masked positions take \(\mathrm{finfo(dtype).min}\), not \(-\infty\) (a fully masked row softmaxes to NaN with −inf, but merely spreads uniformly with finfo.min). The starter docstring elevates this rationale into part of the contract.
- Causal background: Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762) — in autoregressive decoding, position \(i\) cannot attend to \(j>i\); this problem is the additive-mask generalization of that constraint to "kv_offset old tokens already padded to the left of the KV buffer".

## Full derivation and index convention

**Two coordinate systems**: \(i\) is 0-based inside the query window (0..q_len−1); \(j\) is the **absolute** position in the KV buffer (0..kv_len−1); kv_offset counts old tokens cached to the left of the buffer. During decoding, the "current query row" sees exactly everything from the cache start up to itself, so the diagonal lands at \(j=i+\text{kv\_offset}\). The band constraint uses the **relative** coordinate \(r=j-\text{kv\_offset}\): \(i-r<\text{window}\), i.e. at most window−1 KV positions back from the current query, with the diagonal always visible. Keeping j absolute is what lets the causal/window conditions compose naturally with the "old tokens padded on the left" buffer layout.

**Algebraic shape of the four constraints**: causal gives the upper bound \(r\le i\); window gives the lower bound \(r>i-\text{window}\); together they carve a diagonal band of width window. kv_lens / seq_lens are structure-independent column crops / row crops. All on: visible set = diagonal band ∩ valid columns ∩ valid rows — four ANDed conditions, drop whichever is absent, with no exemptions between them.

**Collapse to a triangle at kv_offset=0, pure causal**: \(j\le i\) is exactly the lower triangle including the diagonal, bitwise equal to torch.tril's boolean pattern — the `causal-mask` problem's upper-triangle mask is a special case of this mask. kv_offset=k shifts everything k columns right (row i's visible set is exactly \(j\le i+k\)): the picture of a decoding step where the KV buffer already holds k old tokens and row i of the query window looks at further-right absolute positions than plain causal would allow.

**The numerical accounting of finfo.min**: fp32's finfo.min ≈ \(-3.4028\times10^{38}\). Logits are O(1..100); adding does not overflow (fp32 max 3.4e38), and the gap between a masked and an unmasked score exceeds \(10^{38}\), so softmax turns \(e^{-10^{38}}\) into an underflowed 0 — masked positions get probability exactly 0, numerically indistinguishable from −inf. **Only fully masked rows differ**: a −inf row softmaxes in the \(-\infty/-\infty\) form and outputs NaN, while a finfo.min row keeps finite hugely-negative logits and softmax yields the finite uniform spread \(e^{x_i-\max}/Z\) (\(1/L\)). That is the entire mechanism behind "fully masked rows must not NaN", and the reason HF / flash-attn avoid −inf.

**window is independent of causal**: causal=False with window=w applies only \(i-r<w\) — future positions unrestricted, only "looking too far back" limited, producing a bidirectional local band centered on the diagonal. The judge specifically covers causal=False + window + kv_offset combinations.

**Batch broadcasting semantics**: B = max(len(query lens), len(KV lens), 1); a length-1 sequence replicates along the batch (one dim saved when an encoder shares a single crop across the batch); both longer than 1 must match. A length that is neither 1 nor B — or both >1 and unequal — raises ValueError.

## Key points and common mistakes

1. **j is absolute, i is relative**: both causal and window must place kv_offset on the j side (\(j\le i+\text{kv\_offset}\), \(i-(j-\text{kv\_offset})<\text{window}\)); writing \(j\le i\) or \(i-j<\text{window}\) collapses across every kv_offset>0 cross-check.
2. **The masked value is dtype-dependent**: fp32 ≈ \(-3.4028\times10^{38}\), fp64 ≈ \(-1.7977\times10^{308}\), always \(\mathrm{finfo(dtype).min}\); hardcoding −inf or fp32's min fails the float64 probe.
3. **Visible positions are exactly 0.0**: the mask is added to logits, so visible positions must add nothing; writing 1.0 (boolean thinking) or smuggling in other bias values fails the bitwise equal.
4. **window ≤ 0 is rejected; window still applies with causal=False**: two independent switches; "window only makes sense with causal" is a misreading.
5. **Pad rows/columns are whole-row/whole-column masks**: rows \(i\ge\text{seq\_len}_b\) see no column (diagonal included); columns \(j\ge\text{kv\_len}_b\) are invisible to every row; lens crops never exempt positions from causal/window.
6. **Length-1 broadcasting**: a length-1 lens replicates to every batch; do not read it as "applies to batch 0 only".
7. **Empty dims are legal**: q_len=0 or kv_len=0 returns a valid (B,1,0,kv_len)/(B,1,q_len,0) tensor, no error.
8. **Validation surface**: negative/non-integer q_len, kv_len, kv_offset; window ≤ 0; illegal dtype; negative/out-of-range/mismatched-length lens — all ValueError.

## Differences from neighboring problems

- **vs `causal-mask`**: that problem keeps the triangle of an (L,L) score tensor and sets the upper triangle to float("-inf") (the kv_offset=0 special case, and it masks the scores themselves); this problem is the 4-D **construction** generalized with kv_offset shift + padding crops + window band, whose output is always the two-level 0.0/finfo.min. At kv_offset=0 with no lenses the triangular decision pattern matches that problem bitwise, but the value domain differs (that one keeps original scores below the triangle and −inf above).
- **vs `sliding-window-attention`**: that one implements the **forward pass** of window attention (in-band softmax weighted sums); this problem only builds the mask, no attention math at all.
- **vs `alibi-bias`**: ALiBi is another additive-bias source (a distance-decaying continuous bias, composable with this mask); this problem's additive term has only two levels, 0 / finfo.min — a switch, not a score.
- **vs `kv-cache-update`**: that one manages the write side (appending tokens to the KV buffer); this problem is the read-side visibility picture when the buffer already holds kv_offset old tokens — the write/read pair of a decoding step.

## Cross-check convention and complexity

- Independent recompute oracle: a triple for-loop over (b,i,j) evaluating the four rules explicitly, with length-1 lens broadcast via `bi % len(lens)` — a completely different path from the vectorized broadcast solution. The mask takes only two values, so the judge can compare bitwise with torch.equal directly. Ten full-combination cases (causal × window × lens × kv_offset, including fully visible, whole-batch masked, and seq-broadcast) plus dedicated probes: tril decomposition, offset=3 shift, window=1 strict diagonal, huge-window degeneration, full-lens equivalence, pad rows/columns, float64, zero dims.
- Complexity: \(O(B\cdot\text{q\_len}\cdot\text{kv\_len})\) time and space — one pass ANDing the visibility conditions, one masked_fill; each individual (i,j) decision is O(1).
