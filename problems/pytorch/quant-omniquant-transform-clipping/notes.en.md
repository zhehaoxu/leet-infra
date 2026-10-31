# Solution Notes · OmniQuant's Equivalent Transform and Weight Clipping

## Sources

- **The transcribed source is `models/llama_linear.py` in `Saad-Fahmy/LLM-QAT` (raw HTTP 200, read line by line twice in this session), not the official OpenGVLab OmniQuant.** Its `OmniQuantLinearWrapper` carries the three core lines of the channel-wise equivalent transform - \(s1\) is the per-output-channel absmax of the weight, \(s2\) the per-input-channel absmax of the (mean-shifted) activation, \(s=\alpha s1+(1-\alpha)s2\), the shift is \(\beta(s1-s)/s1\) with **\(\beta\) reusing \(\alpha\)**, and \(T\) is assembled as `diag(s) @ (I - 1*shift^T / n)` together with its approximate inverse; the LWC initialisation line sits near line 91 of the same file (the lower bound is written as `2 * sigmoid(learnable)` and then combined with the raw amplitude bound through `min`, the learnable being initialised to `0.00`).
- **How the official OmniQuant differs (re-checked against OpenGVLab/OmniQuant on 2026-10-30)**: in the paper 2308.13137 and its official repository (`models/transformation.py`, `quantize/quantizer.py`, `quantize/omniquant.py`) the LET is a **per-input-channel geometric interpolation** \(s_j=\mathrm{clip}(s2_j^{\alpha}/s1_j^{1-\alpha},10^{-5})\) plus a shift \(\delta_j=(\max X_j+\min X_j)/2\), with \(X/s\), \(sW\) and a \(\delta W\) bias correction; there is **no \(\alpha\) grid search** (`--alpha` is one float, default 0.5, and every equivalent-transformation parameter is fit by end-to-end gradient descent - the trained names contain `smooth`, `smooth_scale` or `bound_factor`); and LWC there is `xmax = sigmoid(upbound_factor)*xmax`, `xmin = sigmoid(lowbound_factor)*xmin` with `init_value = 4.`, `scale = range/(2**n_bits-1)`, `x_int = round(x/scale)+round(zero_point)` - it has **neither** `min(w, w*2*sigmoid(alpha))` **nor** the `min/ma/ma2/2/1` WSC table. This exercise therefore belongs to the engineering family "equivalent transform + WSC/LWC" as implemented by LLM-QAT, and only the problem statement is normative for judging.
- The clipping lives in two blocks of the same repository's quantization utilities: WSC (an initial threshold looked up from a fixed table of the per-block `bmax` amplitudes) and LWC (the threshold turned learnable, initialised as \(w\cdot 2\sigma(\alpha)\)).
- The \(\alpha\) grid `[0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]` and the "first minimum" selection (`min(range(len(scalars)), key=scalars.get)`) also come from that search code.

## Sources and deliberate simplifications

- **No model, no calibration forward**: `Xc` is handed over already mean-shifted per column, shape \((T,n)\); the upstream `X = X - X.mean(0)` is not redone here.
- **Square layers only**: \(s1\) (per output channel) and \(s2\) (per input channel) can only be convex-combined when they have the same length, and only then are \(T\) and \(T_{\text{inv}}\) both \(n\times n\). Upstream offers nothing to copy verbatim for rectangular layers, so this problem invents no generalisation and raises `ValueError` instead.
- **LWC is exercised at its initial value only**: the gradient steps of the learnable parameters, their bound clamping and the per-block inner training loop are all excluded; `lwc_steps` is therefore a placeholder input that **changes no returned value** (a dedicated case sweeps 0..8 to pin that).
- **`wsc_w` is an input, not a recomputation**: upstream computes the per-block amplitudes with `bmax(W)`; here the caller supplies one table that every \(\alpha\) shares, and `lwc_bound` is likewise determined by that table alone, independent of \(\alpha\) and of the choice.
- **The LWC constant resolves towards the numeric anchor**: the source line reads `alpha = full_like(w, 0.00)`, but \(2\sigma(0)=1\) holds exactly and anchors no constant at all; the quoted `1.0049958...` is the float32 reading of \(2\sigma(\alpha_0)=1.00499996\) at \(\alpha_0=0.01\). The statement therefore takes \(\alpha_0=0.01\) and returns **the scaled candidate bound itself**, \(|w_{\text{sc}}|\cdot 2\sigma(\alpha_0)\): the \(\min(w,\cdot)\) of the source line does not bite at \(\alpha_0\) (the factor exceeds 1) and only engages after training drives \(\alpha\) negative.
- **Source-path re-check record**: `models/llama_linear.py` in `Saad-Fahmy/LLM-QAT` was fetched twice independently during this session, both times returning HTTP 200 and agreeing line by line with the formulas above (the AdaRound problem shares that repository and the same checking method). The path previously recorded as `wiki-xx/OmniQuant/models/llama_patch.py` queries as Not Found through the GitHub API - that repository name now carries LLM-QAT as its description - so it has been corrected to LLM-QAT; the problem statement remains the only normative specification for judging.
- **\(T_{\text{inv}}\) is the pair carrying \(\delta_{ij}\)**: implementing the transcribed \((1+\text{shift}_j)/n/s_i\) literally (an all-ones matrix) makes \(T\,T_{\text{inv}}\) rank-one, so \(\lVert T\,T_{\text{inv}}-I\rVert_\infty\approx 1\) and upstream's `< 0.05` assertion could never hold, nor could \(\alpha=1\) degenerate to a diagonal. With \((\delta_{ij}+\text{shift}_j/n)/s_i\) one gets \(T\,T_{\text{inv}}=I-(\sum_k \text{shift}_k/n^2)\mathbf 1\,\text{shift}^\top\), a deviation controlled purely by the shift **sum**, which makes the assertion meaningful and turns the \(\alpha\in\{0,1\}\) degenerate points into exactly inverse diagonals with `norm_comp` equal to 0. The statement pins this pair as the sole contract.
- **Quantization stops at "per output row, symmetric"**: grouped scales, asymmetric zero-points, real int4/int8 kernels and OmniQuant's residual/low-rank side branch are out of scope; `n_bits` only enters through \(q_{\max},q_{\min}\).

## Key points and pitfalls

**The rank-one factor is indexed by the column.** `outer(ones, shift)` gives \([i,j]=\text{shift}_j\) and \(T_{ij}=s_i(\delta_{ij}-\text{shift}_j/n)\); writing `outer(shift, ones)` yields \(\text{shift}_i\), a different matrix for which \(T\,T_{\text{inv}}\) no longer stays near \(I\).

**The shift divides by \(s1\).** \(\text{shift}=\alpha(s1-s)/s1\) with \(\text{shift}_i=0\) where \(s1_i=0\) (an all-zero weight row) - there \(s_i=(1-\alpha)s2_i\ne s1_i\), so dividing by zero would give NaN and the degenerate branch must be handled explicitly; when \(s_i=0\) (that row all zero **and** that activation column all zero) \(1/s\) is undefined and the input is rejected with `ValueError`.

**The degenerate points are free self-checks**: \(\alpha=0\Rightarrow s=s2,\ \text{shift}=0\); \(\alpha=1\Rightarrow s=s1,\ \text{shift}=0,\ T=\operatorname{diag}(s1),\ T_{\text{inv}}=\operatorname{diag}(1/s1)\). Both make `norm_comp` exactly 0, so any implementation reporting otherwise has misplaced an index or a `/n`.

**`norm_comp` measures the product's deviation, not a matrix norm.** \(\lVert T\,T_{\text{inv}}-I\rVert_\infty\) is the largest absolute entry; it degrades monotonically with \(\alpha\) (in the example \(1.1\text e{-16}\to4.14\text e{-2}\)), so "minimise the loss first, then look at the identity deviation" really are two independent actions.

**The clip threshold is one scalar over the whole table.** \(w_2=\min_i w_i\) uses the **signed minimum** while \(w_3=\max_i|w_i|\) uses the **largest magnitude**; the five candidates follow the source order

```
min -> w2      ma -> w3/4      ma2 -> w3/2      2 -> w2/8      1 -> w3/8
```

Note that `"2"` and `"1"` are **string names** standing for \(w_2/8\) and \(w_3/8\), not "the 2nd / 1st candidate".

**The loss reference is the transformed \(W'\), not the original \(W\).** The transform itself is free; only "clip + per-row integerization" enters the loss. Using \(X_cW^\top\) as the reference punishes every \(\alpha\) at once and misplaces the chosen \(\alpha\).

**Flattening order and ties.** `losses[i * len(choices) + j]` with \(\alpha\) outer; the first minimum wins, so `alphas` may contain duplicates and ties keep the earlier index (example 2 is built for exactly that). Ties between choices work the same way (`min` always precedes `ma2`).

**Row-quantization details.** `scale = row_absmax / qmax`, an all-zero row uses `scale = 1` and code 0 (no `0/0`); rounding is half-to-even; the clamp interval is asymmetric, \([q_{\min},q_{\max}]=[-2^{b-1},\,2^{b-1}-1]\).

**The returned dtype is always float32.** Everything is computed in float64 (otherwise BLAS blocking would let the pinned example digits drift with the thread count), and float64 inputs still return float32 tensors, with `losses`, `norm_comp` and `best_alpha` as Python floats.

## Cross-checking

- Reference implementation: vectorized `torch` in float64; `s1 = W.abs().amax(1)`, `s2 = Xc.abs().amax(0)`, the rank-one factor via `torch.outer(ones, shift)`, \(T\) as `torch.diag(s) @ (eye - ...)`, clipping as `torch.clamp(Wt, -t, t)`, row scales via `amax(keepdim=True)/qmax` with zeros replaced by 1, and the error as `(Xc@Wt.T - Xc@deq.T).pow(2).sum()/(T*n)`; the winner is picked with `min(range(len(losses)), key=losses.get)`.
- Independent oracle: pure-Python nested loops - hand-written \(s1/s2\) maxima, hand-written \(n\times n\) products (three of them), both \(\delta + shift/n\) matrices built element by element, \(w_2,w_3\) recomputed per choice, per-row integerization with `round()` (also half-to-even) and element-wise squared-error accumulation, sharing no control flow with the reference.
- Coverage: example 1 (the joint optimum at \(\alpha=0.25,\ t=1.94\)), example 2 (three alphas tied exactly), the \(\alpha=0/1\) degenerates, an \(s1\) with a zero row, a constant weight (so that \(w_2=w_3\)), each choice run alone to check the threshold table, element-wise checking of the \(2\sigma(0.01)\) constant, a dtype sweep, non-contiguous views, inputs not being mutated, the inertness of `lwc_steps`, invalid inputs (out-of-range alpha / empty grid / mismatched shapes / non-square / unknown choice / out-of-range bits), plus a seeded random sweep.
- Invariants: `norm_comp == max|T@T_inv - I|` recomputed from the returned matrices, `clipped_weight == clamp(T@W@T_inv, -t, t)` with \(t\) rebuilt from `best_wsc_choice`, the minimum of `losses` attained at the reported index, and both \(T\) and \(T_{\text{inv}}\) diagonally dominant.

## Complexity

- Per \((\alpha,\text{choice})\): \(O(n^3)\) for \(T W\) and then \(T W T_{\text{inv}}\), plus \(O(Tn^2)\) for the two \((T,n)\times(n,n)\) projections.
- \(T\) and \(T_{\text{inv}}\) depend only on \(\alpha\), so \(T W T_{\text{inv}}\) can be reused across the choice loop, leaving only \(O(n^2)\) of clip and quantization work inside it.
- Overall \(O(|A|\cdot(n^3+Tn^2)+|A||C|\cdot n^2)\); a real OmniQuant run multiplies this by the number of calibration batches and inner iterations - precisely the training cost this problem excludes.

## Related problems

- `quant-awq-clip-scale-search`: AWQ's joint scale-and-clip grid, diagonal per-channel scaling plus grouped symmetric quantization; this problem swaps the diagonal for a "diagonal + rank-one" equivalent transform and turns the clip threshold into a lookup table.
- `awq-activation-scale` / `quant-smoothquant-alpha-search`: single-parameter search and the "migrate the magnitude to the weight side" accounting, for contrast.
- `quant-hadamard-transform`: another (orthogonal) transform matrix; here \(T\) is not orthogonal and relies on \(T_{\text{inv}}\) to cancel it approximately.
- `group-wise-quantize` / `per-channel-quantize`: this problem only needs the per-output-row symmetric scale; grouping and zero-points live in those problems.
