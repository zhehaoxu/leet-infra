# Speculative Decoding: Cascaded Multi-Draft Residual Filling

## Problem

Rejection sampling for a single draft only handles "one position". In the multi-draft setting (Medusa-2's sequential mode, MTP's stacked drafts) one position may have \(L\) proposal distributions \(q_1,\dots,q_L\) whose masses overlap heavily—the same high-probability token is bet on by all \(L\) drafts. If each level independently applies \(\min(1, p/q)\), the overlap is counted repeatedly and total accepted mass exceeds 1.

The correct bookkeeping is **residual filling level by level**: maintain a vector `rem` of target mass not yet explained, initialized to \(p\). Each draft level can only contribute the overlap with `rem`:

\[
\text{acc}_l = \min(q_l,\ \text{rem}),\qquad \text{rem} \leftarrow \text{rem} - \text{acc}_l
\]

Mass a previous level already consumed is unavailable to later levels. After all \(L\) levels, \(|rem|\) is the probability that no draft covers anything—that part must fall back to the target model:

\[
\text{out} = \sum_{l} \text{acc}_l + |\text{rem}| \cdot p
\]

This exposes an easily missed **bias**: the fallback spreads the residual mass in proportion to \(p\) instead of \(rem/|rem|\). True rejection sampling uses the latter, so \(\sum\text{acc}_l + \text{rem} \equiv p\) and the distribution is preserved exactly; spreading in proportion to \(p\) inflates the tokens the drafts cover (from \(p(x)\) to \((1+|rem|)p(x)\)) and shrinks the ones they miss, tilting the sampling distribution toward wherever the drafts are accurate. This problem outputs the former precisely so the bias becomes visible.

## Task

Implement `solve(p, qs)`:

- Input: `p` (`(T,)` float32, the normalized target distribution); `qs` (`(L,T)` float32, where row \(l\) is the distribution of draft level \(l\), each row normalized, \(L\ge 0\)).
- Outputs:
  - `out` (`(T,)` float32): `Σ acc_l + |rem|·p`;
  - `accepted_mass` (`(L,)` float32): total mass contributed by each level, \(|\text{acc}_l|\);
  - `target_mass` (float32 scalar): \(|rem|\), the fallback probability.
- Compute everything in float64 internally and cast to float32.
- Raise `ValueError` when `p`/`qs` contain negatives, when `p` or any row of `qs` is not normalized (tolerance \(10^{-5}\)), when vocab sizes differ, or on wrong dimensionality.

## Example

```
p  = [0.5, 0.3, 0.2]
q1 = [0.5, 0.5, 0.0]
q2 = [0.0, 0.0, 1.0]

level 1: acc1 = min(q1, p)      = [0.5, 0.3, 0.0]   mass 0.8
         rem  = p - acc1        = [0.0, 0.0, 0.2]
level 2: acc2 = min(q2, rem)    = [0.0, 0.0, 0.2]   mass 0.2
         rem  = rem - acc2      = [0.0, 0.0, 0.0]

out          = acc1 + acc2 + 0 * p = [0.5, 0.3, 0.2]   (= p, full coverage)
accepted_mass= [0.8, 0.2]
target_mass  = 0.0
```

The two levels are exactly complementary: q1 covers the first two tokens and q2 only fills the third one it missed. If q2 also bets on the first token, that mass is **unavailable** at level 2 because `rem` is already 0 there.

## Constraints

- **Every level must subtract from rem**: writing `sum(min(q_l, p))` counts overlapping mass repeatedly and total accepted mass can exceed 1.
- **The accumulation is a min, not a product**: multiple drafts do not "accept simultaneously" but "fill the hole in turn", so it is a per-token take-the-minimum accumulation.
- **`out` is independent of level order**: per token the hit mass equals `min(p(x), Σ_l q_l(x))` (level-by-level min equals a single truncation), so `out` depends only on Σq, not on order. `accepted_mass`, however, depends on order strongly: who goes first decides who gets the credit.
- **The fallback uses \(p\), not the residual**: this is the bias point of the problem and the boundary against strict rejection sampling; using the residual yields `out ≡ p`, and every check would "look more correct" while the semantics changed.
- Allow \(10^{-5}\) tolerance when validating normalization: upstream softmax usually leaves drifts around 1e-7, so requiring exact equality would reject legitimate inputs.
