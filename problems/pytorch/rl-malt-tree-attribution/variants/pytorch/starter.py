"""MALT (arXiv:2412.01928, COLM 2025) -- search-tree credit attribution
ledger.

The judge injects a single ``solve``; dispatch by the ``mode`` keyword:

- mode="propagate": value iteration over the 3-level G→V→R tree
  (kwargs: rewards as an n x n x n binary list [j][k][l], n)
  -> (verifier_values n x n, generator_values n); both levels are the
  sample mean over direct children, no discount factor
- mode="binarize": strict threshold labels (kwargs: values, theta=0.5)
  -> V > theta is correct, V == theta is incorrect (majority-vote
  alignment)
- mode="pairs": same-parent preference pairs (kwargs: unit_values,
  parent_ids, theta) -> (pos_index, neg_index) pairs, positive subset x
  negative subset within each parent, parents in ascending order
- mode="sft": positive-trace collection (kwargs: unit_values,
  parent_ids, theta) -> {positives (input order), counts per parent}
- mode="attribute": full per-question pipeline (kwargs: rewards, n,
  theta) -> verifier/generator values and labels, flat refiner
  values/labels (linear index j*n*n+k*n+l), D_R / D_V / D_G pairs, and
  the per-role SFT ledger (the generator's D_G pairs are constructed
  but SFT-only per the paper)

No official code exists for MALT; the oracle is the paper formula
itself (unique deterministic arithmetic): V(r)=leaf binary reward,
V(v)=E_l[V(r)], V(g)=E_k[V(v)], binarize at >0.5, and preference pairs
grouped by the shared parent context (never across branches).
"""
