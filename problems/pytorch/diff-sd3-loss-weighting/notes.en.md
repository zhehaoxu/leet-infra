# Deep Dive: SD3 Sigma Weighting and Confidence-Aware Cross-Entropy

## Where it comes from

HuggingFace diffusers' `src/diffusers/training_utils.py`:

- `compute_loss_weighting_for_sd3(weighting_scheme, sigmas)` — the roster that the SD3 training script uses to weight each sample's flow-matching error by σ. The SD3 paper is arXiv:2403.03206.
- `compute_confidence_aware_loss(logits, labels, *, lambda_conf, temperature, per_token_weights, ignore_index)` — a combined loss for token-classification heads: plain CE plus an entropy penalty applied only on tokens that are already predicted correctly.

**Source correction**: the backlog once described the roster as four schemes ("sigma_sqrt / cosmap / linear / quadratic"), but the current upstream (`main`, checked line by line) has only three branches — `sigma_sqrt`, `cosmap`, and the `ones_like` fallback. There is no `linear` or `quadratic` branch. This problem ships only the three that actually exist and invents no roster entries.

## Key ideas and pitfalls

1. **The `.float()` inside `sigma_sqrt`**: upstream writes `(sigmas**-2.0).float()`. A float64 σ goes in and a float32 weight comes out, while `cosmap` and `ones_like` skip `.float()` and follow the input dtype. "Unifying the dtypes for tidiness" is exactly the change that breaks this semantics.
2. **Roster matching is strict string equality and has no raising `else`**: `"none"`, the empty string, `"cos_map"` and `"SIGMA_SQRT"` all fall silently into `ones_like`. Distributed training scripts rely on that fallback; turning it into an exception would break upstream callers.
3. **`sigma_sqrt` arithmetic in float32**: `0.1**-2.0` evaluates to `99.999992` in float32, not `100`. The example keeps that value precisely because it shows there is no float64 intermediate in the weight chain.
4. **The shape of `cosmap`**: `bot = 1 - 2σ + 2σ² = 2(σ-1/2)² + 1/2`, symmetric about σ=1/2, minimal at the midpoint (giving the maximal weight \(4/\pi\)), and returning to \(2/\pi\) at both σ→0 and σ→1.
5. **The `lambda_conf == 0` short circuit**: upstream skips the entropy entirely and returns `(loss_sft, loss_sft, 0)`. The first return value is *exactly* the second and the third is exactly 0; an implementation that computes the entropy anyway and adds zero may still disagree bit-wise because of a different float path.
6. **Temperature only touches the entropy**: the CE sees the raw logits, and the temperature only enters `softmax(logits.float()/temperature)`. Feeding the temperature into the CE as well changes `loss_sft` — the most common mistake here.
7. **Where `logits.float()` loses precision**: the entropy is always computed in float32 and then cast back with `.to(logits.dtype)`. For float64 logits `loss_conf` therefore carries only ~1e-7 relative accuracy, so the oracle comparison needs `1e-6`.
8. **Both denominators clamp at 1**: `loss_sft` divides by `Σ weights·valid` and `loss_conf` by `Σ weights·correct`. With every weight at 0.25 and two valid tokens the denominator is 0.5, clamped to 1, so the loss becomes "0.25 times the sum of the two terms" rather than their average. Zero-weight positions enter neither numerator nor denominator.
9. **`ignore_index` may be a legitimate class id**: it is a plain integer comparison and need not be negative. With `ignore_index=0` and label 0, a position is excluded wholesale even when the argmax hits — that is what the `valid &` inside `correct` is for.
10. **Argmax tie semantics**: `torch.argmax` returns the first maximal index, so the hand-written `_argmax` in the oracle must compare with a strict greater-than to agree.

## How the judge cross-checks

The independent `_oracle` calls no loss operator at all: per token it computes the CE from a `math.log`/`math.exp` max-trick logsumexp, builds the softmax with python scalars to get the entropy, and hand-writes both denominators and the `max(..., 1.0)` truncation. Nine assertion groups:

- the three weighting schemes and the example loss (`loss`/`loss_sft`/`loss_conf` to 1e-6) plus the `lambda_conf=0` short circuit;
- the dtype matrix: float64/float32 × three schemes, `sigma_sqrt` always float32, `cosmap` following the input;
- roster fallback: five non-exact strings all return ones without raising;
- `cosmap` symmetry, the \(4/\pi\) maximum at the midpoint, the \(2/\pi\) endpoints, and a 19-point grid comparison;
- `ignore_index` exclusion (including `ignore_index=0` as a legitimate class id) with a positive control for hits;
- the confidence term counting only hit positions (misses excluded, degenerate to 0 when nothing hits);
- the temperature changing only the entropy: `loss_sft` compared with `torch.equal` across four temperatures, `loss_conf` strictly increasing;
- denominator truncation: weights 0.25×2 → 0.25·Σnll, zero-weight positions dropped, weight dtype following the logits;
- a bit-wise comparison over two (V,L,B) shapes × 3 λ values × 3 temperatures × {no weights, random weights}, plus 22 invalid-input cases.

## Complexity

The weighting part is an \(O(|\sigma|)\) element-wise pass; the loss part is one CE plus one softmax plus two reductions over \(\prod(\cdots,V)\) elements, with no extra buffers.
