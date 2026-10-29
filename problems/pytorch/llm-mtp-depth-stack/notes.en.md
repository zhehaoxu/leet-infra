# Deep Dive: Multi-Token Prediction Depth Stack

## Where it comes from

§2.2 of the DeepSeek-V3 technical report (arXiv:2412.19437) describes Multi-Token Prediction: \(D\) MTP modules sit on top of the backbone, and module \(k\) computes

\[ h'^{k}_i=M_k\big[\ \mathrm{RMSNorm}(h^{k-1}_i)\ ;\ \mathrm{RMSNorm}(\mathrm{Emb}(x_{i+k}))\ \big] \]

with \(M_k\in\mathbb R^{d\times 2d}\); the output goes through the `OutHead` shared with the backbone to predict \(x_{i+k+1}\), and training adds \(\sum_k \lambda_k\,\mathcal L^{k}_{\text{MTP}}\) to the main loss.

This problem keeps only the \(M_k\) stage (normalize, concatenate, project) and leaves out the shared head, embedding tying, and loss weighting. The RMSNorm here is the **non-affine** version — no \(\gamma\) is supplied; adding one back just means a per-dimension multiply after normalization and changes none of the structural properties. That choice is what lets "is the hidden state really normalized?" and "is the embedding normalized separately?" be asserted directly via **positive-rescaling invariance**.

## Key ideas and pitfalls

1. **The depths are serial, not parallel**: depth \(k\) consumes \(h^{k-1}\), not the backbone's \(h^0\). Feeding \(h^0\) to every depth degenerates into \(D\) independent one-step predictions, and the assertions in `check_depth_chaining` ("changing depth 2 changes \(h^2\)", "changing depth 1 cascades into \(h^2\)") catch it immediately.
2. **The embedding is shifted by one per depth**: depth \(k\) reads `embeddings[:, k:k+S]`. Passing `embeddings[:, :S]` (every depth reading the same slice) is the most common off-by-one here — note both variants have identical shapes and run without any error.
3. **The second input is the token embedding, not the previous prediction**: MTP trains with teacher forcing and reads from the shared embedding table. Feeding self-produced tokens changes the gradient path completely.
4. **Normalize the halves separately, then concatenate**: concatenating first and normalizing afterwards couples the two scales (the norm gap between \(h\) and the embedding gets averaged away). `check_rms_scale_invariance` rescales `hidden` and the embedding slice independently precisely to separate the two normalizations.
5. **Concatenation order is tied to \(W\)'s row order**: the convention is `[RMS(h) ; RMS(Emb)]`, so the first \(d\) rows of \(W\) multiply the hidden state and the last \(d\) the embedding. Swapping them keeps every shape valid while getting every number wrong; the oracle comparison catches it value by value.
6. **eps goes inside the sqrt**: \(\mathrm{RMS}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\), not \(\sqrt{\mathrm{mean}(x^2)}+\varepsilon\) and not \(x/(\sqrt{\mathrm{mean}}+\varepsilon)\). The three agree for tiny eps and diverge for large ones, so the judge cross-checks once with `eps=1e-2`.
7. **The embedding at position 0 must never be read**: that token is already absorbed into \(h^0\). This is directly assertable — perturb `embeddings[:, 0]` and every depth's output must stay put.

## How the judge cross-checks

The reference solution loops over depth in float64: `_rms` uses `mean(dim=-1)` plus `sqrt`, the concatenation is projected with `einsum("bsf,fe->bse")`, and each state is cast to float32 and collected. The `_oracle` reuses no tensor ops: it pulls out each `(batch, position)` pair, computes the mean square, the square root, the normalization, the concatenation, and the multiply-accumulate with python scalars, then rebuilds a tensor. Five structural assertion groups: positive-rescaling invariance of both normalizations, the serial depth dependency (depth 1 affects both states, depth 2 only the second), position sensitivity of the embedding windows (positions 0, 1, and \(S+1\)), oracle comparison across three shapes and multiple eps values, and invalid inputs. Tolerance `allclose(atol=1e-5)`.

## Complexity

Two RMSNorms (\(O(BSd)\)) plus one linear (\(O(BS\cdot 2d\cdot d)\)) per depth, for \(D\) depths: time \(O(2DSBd^2)\); apart from the outputs only the current hidden state is kept, \(O(BSd)\).
