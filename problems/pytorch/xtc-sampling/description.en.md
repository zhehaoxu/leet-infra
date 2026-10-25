# Sampling: XTC Mid-Probability Elimination (Exclude Top Choices)

## Problem Statement

top-k, top-p and min-p all **cut the tail**: they clear out the low-probability long tail and leave the head
untouched. The consequence is that whenever the model is extremely confident at some step (say
\(p_{\max}=0.98\)), the top two candidates **monopolize** the whole distribution and the output keeps orbiting
the same few high-frequency words. **XTC** (Exclude Top Choices, an implementation technique from llama.cpp and
text-generation-webui) points that knife **the other way**: it ignores the tail and looks at "the contiguous head
whose probability is absurdly high". Given an absolute probability threshold \(t\), it removes every candidate
with \(p\ge t\) in the descending order **except the last one of them**, keeps that boundary candidate plus the
entire tail, and renormalizes. Probability mass from the middle and the tail therefore comes back into play and
the output is forced "one level down", without falling into the random long tail.

Two points of the specification deserve to be stated up front. First, "at least two head candidates must pass
before any cut happens": if only the single most probable candidate clears the threshold
(\(\text{pos\_last}=0\)), the removal set is empty and XTC does nothing — this guard makes XTC an identity
transform when the model truly believes in only one token. Second, XTC is a cut in **probability space** (the
comparison happens on \(p\) **after** the softmax, with \(t\le 0.5\)); it is not a logit-domain criterion and uses
no \(\mu/\sigma\) statistics at all, which separates it cleanly from `top-n-sigma-sampling` in the same
collection. Finally, XTC fires **stochastically**: each row first passes a Bernoulli-style probability gate that
decides whether this step cuts at all (\(u_{\text{gate}}\le\) `probability`), so this problem has **two**
randomness entry points and both must be supplied as explicit arguments.

## Rules

Let \(V\) be the per-row vocabulary size, \(t\) the threshold, \(\pi\) the trigger probability and `min_keep` the
survivor floor, with \(\pi\in[0,1]\) and \(t\in(0,0.5]\). All operations run **row by row, independently**, along
the last dimension.

- **Early exit (any one of these makes the row a no-op)**: \(\pi\le 0\); or \(t>0.5\) (then at most one candidate
  can pass, so a cut is meaningless and the sampler switches itself off entirely); or \(V<2\) (with fewer than two
  candidates the "keep the last one" rule does not even exist). On early exit `filtered` is exactly
  \(\operatorname{softmax}(z)\) and `applied` is `False`, but `chosen` is still sampled with \(u\).
- **① Probability gate.** Take the explicit argument \(u_{\text{gate}}\) per row; the row may only fire when
  \(u_{\text{gate}}\le\pi\) (this is the llama.cpp line that returns as soon as `chance > probability`).
- **② Descending ranks.** With \(p=\operatorname{softmax}(z)\), order the columns by **descending** probability,
  breaking ties by **ascending original index**, giving the original indices at ranks \(0,\dots,V-1\). Softmax is
  monotone ⇒ the candidates above the threshold form a **contiguous prefix** of that descending array.
- **③ Prefix boundary.** Let
  \[
  cnt=\#\{\text{candidates in the descending contiguous prefix with } p\ge t\},\qquad \text{pos\_last}=cnt-1 ,
  \]
  i.e. "the rank of the last candidate that still passes" (equivalent to llama.cpp's "break at the first
  \(p<t\)"). If \(cnt=0\) then \(\text{pos\_last}=-1\); \(\text{pos\_last}=0\) exactly when \(cnt=1\).
- **④ Removal.** The cut happens **if and only if** all three hold:
  \[
  u_{\text{gate}}\le\pi\quad\wedge\quad \text{pos\_last}>0\quad\wedge\quad V-\text{pos\_last}\ge \texttt{min\_keep}.
  \]
  Then the tokens at ranks \([\,0,\ \text{pos\_last}\,)\) are removed (**strictly left-closed, right-open** —
  rank \(\text{pos\_last}\) itself **survives**), and the whole tail of ranks \([\text{pos\_last},\,V)\) survives.
  The third clause is the survivor floor: if only \(V-\text{pos\_last}\) candidates would remain and that is below
  `min_keep`, the entire cut is abandoned.
- **⑤ Renormalization.** `filtered` \(=\;p\odot\text{keep}\,/\,\textstyle\sum p\odot\text{keep}\), kept in
  **original vocabulary order**, with removed positions exactly \(0\) and each row summing to \(1\).
- **⑥ Sampling.** Inverse-transform the explicit \(u\) on the **original-order** CDF of `filtered`:
  \[
  \text{chosen}=\min\{\,i:\ \textstyle\sum_{j\le i}\text{filtered}_j\ \ge\ u\,\},
  \]
  falling back to the last index if no CDF entry reaches \(u\) (a floating-point edge case). Under this
  definition \(u=0\) always returns index \(0\), **even if that position carries zero mass** (removed positions
  are legal return values) — this is the fixed convention of the problem.
- **Return the 3-tuple** \((\text{filtered},\ \text{chosen},\ \text{applied})\): `filtered` has shape `(..., V)`,
  `float32`; `chosen` has shape `(...,)`, `int64` (original vocabulary index); `applied` is a **boolean tensor**
  of shape `(...,)` marking whether the cut actually happened in that row.

## Example

```
Input: z = [3.0, 2.5, 2.0, 0.0, -1.0, -2.0] (V=6), π = 1.0, t = 0.05, min_keep = 0,
       u_gate = 0.5, u = 0.4
  p = softmax(z) = [0.487983, 0.295977, 0.179519, 0.024295, 0.008938, 0.003288] (already descending)
  contiguous prefix with p ≥ 0.05 = ranks 0,1,2 → cnt = 3 → pos_last = 2
  gate: 0.5 ≤ 1.0 ✓    pos_last > 0 ✓    survivors V − pos_last = 4 ≥ min_keep 0 ✓  → fires
  remove ranks [0, 2) = rank 0 and rank 1 (token 0, token 1); keep rank 2 (token 2, "the last one that passed")
  + the tail 3,4,5
  kept mass = 0.179519 + 0.024295 + 0.008938 + 0.003288 = 0.216040
  filtered = [0.000000, 0.000000, 0.830953, 0.112457, 0.041371, 0.015219]
  CDF      = [0.000000, 0.000000, 0.830953, 0.943410, 0.984781, 1.000000]
  u = 0.4 → the first entry ≥ 0.4 sits at position 2
Output: chosen = 2, applied = True
        (what got zeroed is the two MOST probable head candidates, while tail token 5 still keeps 0.015219)
```

```
Same z, only min_keep changes: the survivor count 4 is a hard constraint
  min_keep = 3 → 4 ≥ 3 ✓           applied = True, chosen = 2
  min_keep = 4 → 4 ≥ 4 ✓ (exactly passes) applied = True, chosen = 2
  min_keep = 5 → 4 ≥ 5 ✗           applied = False, filtered is the plain softmax (nothing was cut)
```

```
Two boundaries: "at least two head candidates" and the fully-degenerate case
  Only one head candidate passes: z = [9.0, -1.0, -2.0, -3.0], π = 1.0, t = 0.5, min_keep = 0,
                                 u_gate = 0.1, u = 0.1
    p = [0.999932, 0.000045, 0.000017, 0.000006] → cnt = 1 → pos_last = 0 → no cut
    applied = False, filtered = p (XTC never leaves the argmax standing entirely alone)
  Every candidate passes: z = [0.0, 0.0] (p = [0.5, 0.5]), t = 0.5, π = 1.0, min_keep = 0,
                         u_gate = 0.0, u = 0.5
    cnt = 2 → pos_last = 1 → remove rank 0, survivors 2 − 1 = 1 ≥ 0 → applied = True
    filtered = [0.000000, 1.000000], CDF = [0.000000, 1.000000], chosen = 1
    (the argmax is cut away, and this step becomes a deterministic pick: every u in [0,1) yields 1)
```

```
Batched: the gate opens or closes per row
  z = [[3.0, 2.5, 2.0, 0.0, -1.0, -2.0],
       [0.2, -0.1, 1.4, -2.0, 0.5, 0.1]], π = 0.5, t = 0.05, min_keep = 0,
  u_gate = [0.2, 0.9], u = [0.3, 0.7]
  row 0: 0.2 ≤ 0.5 ✓ → same removal as above → filtered[0] = [0.000000, 0.000000, 0.830953, 0.112457,
         0.041371, 0.015219], first CDF entry ≥ 0.3 is at position 2
  row 1: 0.9 > 0.5 ✗ → gate closed, identity transform → filtered[1] = [0.134654, 0.099754, 0.447067,
         0.014920, 0.181764, 0.121840], CDF = [0.134654, 0.234408, 0.681476, 0.696396, 0.878160, 1.000000],
         first entry ≥ 0.7 is at position 4
Output: applied = [True, False], chosen = [2, 4]
```

## Constraints

- **CPU only, everything hand-written.** The grading environment `.venv-eval` contains only `torch` / `numpy` /
  `transformers` and is **offline**: no llama.cpp, no HF sampler / `LogitsProcessor` classes to borrow, no
  `sklearn` or `nltk` either — both the removal and the inverse-transform sampling must be written yourself.
- **All randomness must come from explicit arguments**: this problem has **two** independent entry points — the
  probability gate \(u_{\text{gate}}\) (deciding whether this step cuts) and the lottery draw \(u\) (deciding who
  among the survivors is picked). Both are uniform random numbers in \([0,1)\) with shape `(...,)` (one per row;
  a scalar is accepted when there is a single row). Do **not** call `torch.rand` / `manual_seed`, and do not let
  \(u_{\text{gate}}\) and \(u\) reuse the same number.
- Shapes and dtypes: input `logits` is `(..., V)` `float32`; outputs `filtered` is `(..., V)` `float32`,
  `chosen` is `(...,)` `int64`, `applied` is `(...,)` `bool`.
- Ranges: \(\pi\in[0,1]\) (switched off when \(\le 0\)); \(t>0\), and the whole sampler is off when \(t>0.5\);
  `min_keep` is a non-negative integer; \(u_{\text{gate}},u\in[0,1)\).
- **Identities that always hold**: any of the three early-exit conditions, a closed gate,
  \(\text{pos\_last}\le 0\), or too few survivors — all four cases give `applied = False` with `filtered` equal
  elementwise to \(\operatorname{softmax}(z)\); when the cut fires, each row of `filtered` still sums to \(1\) and
  the removed positions are **exactly** 0 (not merely close to 0).
- Grading tolerances: `filtered` with `atol=1e-5, rtol=1e-5`; `chosen` and `applied` must compare **exactly equal**.
- Division of labour with neighbouring problems (details in `notes.md`): `top-n-sigma-sampling` uses \(\mu/\sigma\)
  to **keep** a statistically upper band; `min-p-sampling` / `top-p-sampling` / `top-k-sampling` only ever
  **cut the tail** and never touch the head; this problem is the one sampler whose cut direction is flipped around
  and which specifically **removes the high-probability head prefix**.
