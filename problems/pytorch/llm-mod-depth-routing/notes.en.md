# Deep Dive: Per-Layer Routing in Mixture-of-Depths

## Where it comes from

- Raposo et al., *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*, DeepMind, 2024, arXiv **2404.02258**. The central equation (1) is:

  \[
  x_i^{l+1}=
  \begin{cases}
  r_i^l\, f_i(\tilde X^l)+x_i^l, & r_i^l > P_\beta(R^l)\\
  x_i^l, & r_i^l < P_\beta(R^l)
  \end{cases}
  \]

  The router is a scalar \(r_i^l=w_\theta^\top x_i^l\), the threshold \(P_\beta\) is the \(\beta\)-quantile of the router weights with \(\beta=1-C/S\) (\(C\) capacity, \(S\) sequence length), so "above the quantile" means "the top \(C\)". \(\tilde X^l\) holds only the selected tokens. The auxiliary loss is a binary cross-entropy on the router outputs.
- Note: an earlier version of the backlog listed this as 2202.07736, which is wrong; 2404.02258 is authoritative here, verified against the title and authors when the problem was written.
- Division of labour with the `moe` collection: MoE combines along the **expert axis** with softmax/top-k over experts, whereas this problem prunes along the **depth (layer) axis**. Each layer has a single scalar router vector and no mixture of expert weight matrices. No other problem in the library does depth routing.

## Core points and pitfalls

1. **The threshold is a capacity, not an absolute value**. The paper uses the \(\beta\)-quantile, which is equivalent to "take top-k with k=min(C,T)"; the two differ only on ties, and the paper does not define tie behaviour. This problem pins it to (score descending, index ascending) so the same input always produces the same `route_mask`.
2. **The router score multiplies the block output**, it is not a binary switch. Dropping the \(s_i\) factor turns every routing gain into coefficient 1, and `check_single_token_block_scaling` (capacity 1 gives \(x\leftarrow x+s\cdot xW\)) fails immediately.
3. **The block only sees the subsequence**. It is not "attention over the whole sequence, then keep the selected rows" — that would leak dropped tokens into the retained tokens' attention. `check_block_scope_selected_only` perturbs an unselected token and asserts zero influence on the others.
4. **The subsequence preserves order**. Selected indices must be gathered in ascending original order so the causal mask matches the original token order. Gathering in score order makes a high-scoring token the first subsequence row, hiding the tokens that should precede it; `check_gather_preserves_order` builds a case where score order differs from index order.
5. **Routing uses the current state**. Layer \(l+1\) scores the state written back by layer \(l\), not the initial input. `check_router_uses_current_state` compares the two-layer case where "current" and "initial" states select different tokens.
6. **Unselected tokens stay exactly unchanged** — bitwise equal, not "multiplied by zero with a floating-point residue"; `check_unselected_untouched` asserts with `torch.equal`.
7. **Within a layer the semantics are snapshot-based**: the block is applied to the whole subset at once, so later rows of the subset must not observe the write-back of earlier rows. Gathering a copy first and writing back once satisfies this naturally.
8. **Edges**: capacity 0 computes nothing but still reports scores; capacity \(\ge T\) selects everything, degenerating to ordinary causal self-attention with score scaling. `capacities` rejects `bool`/`float` because `True` would silently act as 1.
9. **The block is a stand-in**: a real MoD block is a full attention+MLP stack; here it is a shared-q/k/v causal self-attention with 1/√d scaling, chosen so that order sensitivity inside the subsequence is judgeable and reproducible without any real weights.

## How it is judged

- `_oracle` reimplements the loop in numpy: `sorted(..., key=lambda i: (-s[i], i))` for capacity, `sorted()` for the order-preserving gather, and a hand-written row-by-row causal softmax — no code shared with the vectorized solution.
- Ten `check_*` functions: the example, capacity and ties, single-token scaling, untouched unselected rows, block scope, order-preserving gather, current-state routing, the return contract (shapes/dtypes/scores as dot products), five randomized oracle configs including capacity 0 and over-capacity, and 22 invalid inputs.
- Mutation testing (ascending selection, gather in score order, block over the whole sequence, missing router multiplier, stale router state, no causal mask, missing 1/√d, capacity plus one, misaligned subset rows) is caught by at least one check.
- Both float32 and float64 inputs must work; for float32 inputs the oracle is run on the inputs upcast to float64 so the routing order is fixed (otherwise the two precisions could rank scores differently).

## Complexity

- Per layer: scoring \(O(Td)\), subsequence projection \(O(cd^2)\), causal attention \(O(c^2d)\), write-back \(O(cd)\), with \(c=\min(C_l,T)\).
- Total time \(O\!\left(\sum_l (Td+c_l d^2+c_l^2 d)\right)\), extra space \(O(Td+c\,d)\), plus \(O(LT)\) for `route_mask` and `router_scores`.
