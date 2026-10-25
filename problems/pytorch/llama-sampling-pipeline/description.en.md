# Sampling: llama.cpp Full Pipeline Stage Ordering

## Problem

Individual sampling operators (top-k, top-p, min-p, DRY, repetition penalties...) each get their own problem in this collection; but in a real inference engine they are **chained together**, and the chain's **stage ordering is itself behavior**: llama.cpp puts temperature **after all truncations** (truncation probabilities come from normalizing untempered logits), while HuggingFace puts temperature first — the same top-p has different surviving sets under the two conventions. This problem asks you to assemble llama.cpp's full sampling pipeline in a fixed order:

\[\text{bias(+banned)}\to\text{penalties}\to\text{dry}\to\text{top-k}\to\text{top-p}\to\text{min-p}\to\text{xtc}\to\text{temp}\to\text{dist}\]

and record the **surviving set after every stage**. Randomness is injected via explicit arguments (`u_xtc` drives xtc's coin, `u` drives the final inverse-transform sample), keeping judging deterministic. CPU, float32, no RNG calls anywhere.

## Rules

- The exact signature, argument names and defaults live only in the starter comments and are the single source of truth. `history` is the generated token sequence (a list); the other scalars/tensors are as in the starter.
- **Return the triple `(masks, probs, chosen)`**:
  - `masks`: bool tensor (8, V). Row indices are fixed `0=bias 1=penalties 2=dry 3=top-k 4=top-p 5=min-p 6=xtc 7=temp`; row \(i\) = the surviving set after stage \(i\) ran; banned tokens are `False` from row 0 onward.
  - `probs`: (V,) float — the distribution used by the dist step: normalized over the surviving set, zeros elsewhere.
  - `chosen`: scalar int64. Inverse-transform of `u` over the CDF of `probs` in **original order** (the first index whose prefix sum is \(\ge u\); equality hits); fallback is the last survivor in descending order.
- **Stage semantics** (window arguments share one convention: `last_n<0` takes the whole history, `=0` disables, `>0` takes the trailing \(n\) tokens):
  1. **bias**: each entry of `bias_tokens` **adds** the matching `bias_values` to that logit (out-of-range ids ignored); `banned` hard-masks (logits unchanged).
  2. **penalties**: count within the window; for every alive token with count \(c>0\): multiply the logit \(x\le0\) by `penalty_repeat`, divide \(x>0\) by it, then subtract \(c\cdot\text{penalty\_freq}+\text{penalty\_present}\). Short-circuits entirely when `penalty_repeat=1, freq=0, present=0`.
  3. **dry**: on the **reversed window**, for each offset \(k\) find the longest common prefix with the head (repeats must sit flush against the window's end); only lengths \(rl\ge\) `dry_allowed_length` register, keeping per token only the **strictly longest** repeat; hit tokens have their logit reduced by `dry_multiplier · dry_base^(rl−al)`. Off when `multiplier=0`, `base<1`, or the window is empty.
  4. **top-k**: when `top_k>0` and more than \(k\) survive, keep the top \(k\) by descending logit (ties to the smaller index).
  5. **top-p**: when `top_p<1` and more than 1 survives, accumulate descending probabilities normalized over the **current survivors**, cutting where the sum first reaches \(\ge p\) with at least `min_keep` kept. Under `temp_first=False` these probabilities are **untempered**.
  6. **min-p**: when `min_p>0`, the logit-domain floor is max logit \(+\ln(\text{min\_p})\); scan descending and stop at the first logit **below** the floor once at least `min_keep` are kept.
  7. **xtc**: fires when `xtc_probability>0`, `xtc_threshold\le0.5`, at least 2 survive, and `u_xtc\le\text{xtc\_probability}`: among the descending **head prefix** with probability \(\ge\) the threshold, cut everything except the **last** qualifying token (the tail survives), with survivors still \(\ge\) `min_keep` after the cut.
  8. **temp**: `temp>0` divides each surviving logit; `temp\le0` is greedy — keep only the max-logit token. **Timing** is set by `temp_first`: `False` (llama.cpp convention) after xtc, before dist; `True` (HF convention) right after dry, before all truncations — in that case top-p/min-p/xtc see tempered probabilities, and row 7's snapshot is "temp just ran, truncations have not".
- Every truncation stage's `min_keep` is a **backstop**: cut less or not at all rather than drop below `min_keep` (top-k excepted — under the llama.cpp convention it is not protected by `min_keep`).
- Everything is float32 on CPU and deterministic; ready-made samplers are forbidden.

## Example

`logits=[5,4,3,2,1,0,-1,-2,-3,-4]`, history `[3,4,5,3,4,5]`, `bias={2:+0.7}`, `banned={0}`, `penalty_repeat=1.1, penalty_freq=0.2, penalty_present=0.1, penalty_last_n=-1`, `dry_multiplier=0.8, dry_base=1.4, dry_allowed_length=2, dry_penalty_last_n=-1`, `xtc_probability=0, xtc_threshold=0.05`, `min_keep=1`, `top_k=8`, `top_p=0.6`, `min_p=0`, `temp=0.5`, `temp_first=False`, `u_xtc=0.5`, `u=0.55`.

- Row 0: token 0 banned → 9 survive; token 2's logit becomes 3.7.
- Row 1: tokens {3,4,5} each appear twice; positive logits divided by 1.1 then minus \(2\cdot0.2+0.1=0.5\): token 3 → 1.318, token 4 → 0.409, token 5 → −0.5.
- Row 2: DRY finds token 3 repeating flush against the window end with length 3 ≥ 2 → minus \(0.8\cdot1.4^1=1.12\) → token 3 = 0.198. Still 9 alive.
- Row 3: top-k=8 drops the lowest token 9 → 8 alive.
- Row 4: untempered descending cumulative probability (token 1 ≈0.552, adding token 2 reaches ≈0.961) → cut where the sum first reaches \(\ge0.6\), keeping 2.
- Rows 5/6: min-p=0 and xtc off → unchanged. Row sums `[9, 9, 9, 8, 2, 2, 2, 2]`.
- Row 7: temp=0.5 turns the two survivors' logits into 8 and 7.4; the final distribution is `probs[1]=0.645656, probs[2]=0.354344`, zeros elsewhere; `u=0.55` lands inside token 1's CDF segment → `chosen=1`.

```
masks row sums: [9, 9, 9, 8, 2, 2, 2, 2]
probs:          [0.0, 0.645656, 0.354344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
chosen:         1
```

## Constraints

- `masks` compared with `torch.equal`, `probs` with `atol=rtol=1e-5`, `chosen` with `torch.equal`
- Two discriminating criteria the judge asserts directly: under the **same parameters**, the top-p surviving set differs between `temp_first=False` and `True` (llama's truncations see untempered probabilities — heat does not amplify truncation), and under the llama convention the temp row's survivors are a subset of the xtc row's (temp is the last truncation); with **every knob off**, `probs` equals the softmax of the raw logits
