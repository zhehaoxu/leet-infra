# 4-D Additive Attention Mask Construction

## Problem

An additive mask is a 0 / huge-negative tensor **added** to attention logits — the uniform way to express KV-cache decoding, padding crops, and sliding-window constraints. This problem builds a 4-D additive mask of shape (B,1,q_len,kv_len): **visible positions are exactly 0.0, masked positions exactly \(\mathrm{finfo(dtype).min}\)** — deliberately not \(-\infty\) (see Rules). Coordinate convention: \(i\) is the 0-based position inside the query window, \(j\) is the **absolute** position inside the KV buffer, and kv_offset is the number of old tokens already cached to the left of the KV buffer, so the "current query row" diagonal sits at column \(j=i+\text{kv\_offset}\).

Four independently switchable visibility constraints, ANDed together:

\[j\le i+\text{kv\_offset}\ (\text{causal}),\qquad
i-(j-\text{kv\_offset})<\text{window}\ (\text{local band}),\qquad
j<\text{kv\_len}_b,\qquad
i<\text{seq\_len}_b.\]

## Rules

- The full signature, parameter meanings and defaults live only in the starter docstring; it is the single judging standard. The four constraints: causal=True adds \(j\le i+\text{kv\_offset}\); a given window adds \(i-(j-\text{kv\_offset})<\text{window}\) (window must be \(\ge1\), else ValueError); kv_lens adds \(j<\text{kv\_len}_b\); seq_lens adds \(i<\text{seq\_len}_b\). window and causal are **independent**: causal=False drops the causal constraint and leaves the lone band (future positions unrestricted).
- **Masked value finfo.min instead of −inf**: the mask is added directly to logits. A fully masked row (pad query rows, a batch with kv_len=0) plus −inf yields an all-−inf row and NaN softmax; with \(\mathrm{finfo(dtype).min}\) the row's logits stay finite-and-hugely-negative and softmax spreads them nearly uniformly (\(1/L\)), which is far friendlier to training/decoding. This is the common convention of HF transformers / flash-attn additive masks.
- **Batch broadcasting**: B is the max of the given lens lengths, or 1 when all are absent; each lens sequence has length B or 1 — length 1 is replicated along the batch (e.g. a length-1 query-row lens with a length-B KV-column lens: the query-row constraint is copied per batch, the KV-column constraint is per-batch); both given with both lengths >1 must match, else ValueError.
- **Lens semantics**: every query-row lens item \(\le\) q_len and every KV-column lens item \(\le\) kv_len (out-of-range, negative, or non-int-sequence inputs raise); query rows \(i\ge\text{seq\_len}_b\) are masked across the **whole row**, KV columns \(j\ge\text{kv\_len}_b\) across the **whole column**.
- Returns a contiguous (B,1,q_len,kv_len) tensor of the given dtype: visible positions **exactly** 0.0, masked positions **exactly** \(\mathrm{finfo(dtype).min}\); q_len or kv_len of 0 yields a valid tensor with an empty trailing dim. dtype accepts only float32/float64. q_len/kv_len/kv_offset must be non-negative ints (negative or non-integer raises). No RNG anywhere.

## Example

q_len=3, kv_len=5, kv_offset=2, causal=True (1 = visible, 0 = masked; batch 0):

\[\begin{bmatrix}1&1&1&0&0\\1&1&1&1&0\\1&1&1&1&1\end{bmatrix}\]

Row \(i\)'s visible set is exactly \(j\le i+2\): row 0 sees j∈{0,1,2}, row 1 up to j=3, row 2 up to j=4 — the diagonal \(j=i+2\) and everything left of it. Masked positions hold \(-3.4028235\times10^{38}\) (float32 finfo.min).

A window combination (kv_offset=1, causal=True, window=1): each row keeps a single diagonal point, visible set \(j=i+1\):

\[\begin{bmatrix}0&1&0&0&0\\0&0&1&0&0\\0&0&0&1&0\end{bmatrix}\]

## Constraints

- q_len/kv_len/kv_offset non-negative ints, window \(\ge1\), dtype ∈ {float32, float64}, lens items non-negative and within bounds; violations all raise ValueError
- The mask allows exactly two values: 0.0 and \(\mathrm{finfo(dtype).min}\) (fp32 ≈ \(-3.4028\times10^{38}\), fp64 ≈ \(-1.7977\times10^{308}\)); the judge compares bitwise with torch.equal
- Discriminating criteria asserted directly: causal-only with kv_offset=0 reproduces the torch.tril triangular pattern bitwise; kv_offset=3 shifts the diagonal band 3 columns right (clearly different from kv_offset=0); a huge window degenerates bitwise into pure causal; full lenses (KV lens=[kv_len], query lens=[q_len]) are bitwise identical to no lenses; pad query rows and pad KV columns are whole-row/whole-column finfo.min; KV lens=[0,3] fully masks batch 0 — exactly the case where a fully masked row must spread softmax uniformly instead of producing NaN
