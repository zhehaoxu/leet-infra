# Explained: SFT Per-Turn Loss Weights

## Origin
The per-turn / per-sequence loss normalization trick from engineering practice (same lineage as "token-mean vs sequence-mean" and balancing contributions per turn in multi-turn dialogue). Core invariant: make **each turn's contribution to the total loss** equal a hand-set base, instead of growing linearly with that turn's token count; short turns are not drowned out by long ones. This function implements that idea as per-token weights.

## Core Idea and Pitfalls
1. **Turn = maximal in-row contiguous assistant run**: numbering restarts independently at 0 **within each row**, not globally accumulated; it must reset across rows, otherwise `turn_weights` indices are misaligned.
2. **Weights inversely proportional to turn length**: each token in a turn gets \(=c_t/n_t\), and the turn sums to \(=c_t\). Forgetting to divide by \(n_t\) (writing \(c_t\) per token) lets long turns drown short ones again — exactly defeating the point of this problem.
3. **`turn_weights` out-of-range falls back to 1.0**: when the assistant turns in a sample exceed `len(turn_weights)`, the extra turns use base 1.0.
4. **Non-assistant is always 0.0**: system/user/pad are all 0; an assistant-free row is all 0.

## Cross-checking Details
The reference solution uses "scan row by row to find maximal runs → fill the run with `base/count`". The judge's `_oracle` uses **a different structure**: first stamp each token with its "in-row turn id" (incrementing only at the start of an assistant run), then count the tokens per turn id, and finally back-fill each position with `base[turn id]/that turn's count` — "label and count first, then divide" is a different path from the reference's "find runs first, divide within the run", with identical results. It also asserts "each turn sums to its base", "no assistant → all 0", "turn_weights out-of-range fallback", and "multi-row independent numbering", using `allclose(atol=1e-6)`.

## Complexity
\(O(B L)\).
