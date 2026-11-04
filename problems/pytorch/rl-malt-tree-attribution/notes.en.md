# MALT Implementation Notes (791)

Source: arXiv:2412.01928 v3 (COLM 2025) sec 4.2 (data generation & credit assignment) and A.6.1 (formalization). **No official code release** (survey table lists Code: -; no code link on the arXiv page; the third-party Nimitkothari/MALT is a Groq/LangChain replica and does not pin the semantics). This problem's oracle is the paper formula's unique deterministic arithmetic; the sandbox double-checks it via an independent NumPy translation (per-level np.mean(axis)) plus hand anchors.

## Per-entry semantics

### propagate_values

- Tree encoding `rewards[j][k][l]` = the binary reward of r_{j,k,l}. `V(v_{j,k}) = sum_l / n`, `V(g_j) = sum_k V(v_{j,k}) / n`.
- Paper anchors: 'V(v_{i,j,k}) = E_l[V(r_{i,j,k,l})] ≈ (1/n)Σ_{l=1}^n V(r_{i,j,k,l})', 'V(g_{i,j}) = E_k[V(v_{i,j,k})]'; appendix A.6.1's equivalent form V(g)=1/n²ΣΣV(r).
- **No discount factor**: the tree is 3 deep and the formulas carry no γ. The old candidate description's 'depth discount' was removed after sourcing.
- Shape check: len(rewards)==n and every level's len==n, else ValueError (the branching factor is an explicit input; paper experiments use n=3 → 27 trajectories per question).

### binarize

- Strict `value > theta`. Paper anchor: 'Nodes with values greater than 0.5 are labeled as correct (✓), and those with values less than or equal to 0.5 are labeled as incorrect (×)'.
- With n=3, V(v)/V(g) ∈ {0, 1/3, 2/3, 1} — 0.5 is unreachable; with n=2, {0, 0.5, 1} — the 0.5 boundary is pinned by TREE_B.

### build_preference_pairs

- The full positive-subset × negative-subset combination within each parent, (positive index, negative index), parents traversed ascending.
- Paper anchors: D_R 'where r⁺ is a correct refinement (V(r⁺)=1) and r⁻ is an incorrect refinement (V(r⁻)=0) for the same verifier input v_{i,j,k}'; D_V 'by comparing outputs under the same generator output g_{i,j}'; D_G 'by comparing outputs under the same query q_i'.
- **Never a cross-branch cartesian product**: positives and negatives under different (j,k) parents do not pair.
- Binarization matches the label criterion (strict > θ) — a 0.5-valued unit lands in the negative subset.

### sft_positives

- ✓ indices preserve input order; per-parent (positive, negative) counts output ascending.
- The per-layer criterion differs (paper): R-layer positives are V(r)=1 (the leaf reward itself); V/G layers use the binarized label (✓) — only g with V(g)>0.5 enters SFT.

### attribute_question

- Linear indices: r leaf (j,k,l) → j·n²+k·n+l; v (j,k) → j·n+k; g → j.
- Assembly order: propagate → flatten (v values + parent ids, r values + parent ids) → binarize (v/g/r) → three pair layers → three sft layers.
- G-layer parent_ids are [0]*n (one question = one parent).

## Sandbox and tests

- /tmp/malt_check.py: NumPy-translation cross-check (tree A with n=3, tree B with n=2) + hand anchors (v00=1.0, v01=2/3, g0=7/9, g1=2/9) + pair counting (tree A's R layer has 8 pairs = (0,1)×2 + (0,2)×2 + (1,0)×2 + (1,2)×2 — note (1,0) is 1 positive × 2 negatives = 2 pairs; the first hand count said 1) + the 0.5 boundary + ValueError.
- Hand-anchor lesson: count pairs per parent as (positives × negatives) — do not estimate from the number of 1s; [1,0,0] is 2 pairs, not 1.
- Tests: 6 check functions; TREE_A/TREE_B embedded, EXPECTED values are hand anchors (the formulas are simple enough that no large JSON dump is needed).

## Mutation checklist (mut791 pending)

Easy-to-miss spots: the mean denominator n vs len (identical for valid trees — pinned via the shape check), the two mean axes (axis=2 vs axis=1, i.e. sum_l vs sum_k), strict > vs >= (pinned by tree B), the positive/negative subset split, the parent grouping key, the pair order (positive first / parents ascending), the linear-encoding coefficient j*n*n, the SFT-positive criterion (V layer label vs value), and the ValueError check.

## Boundaries

- 782 `to-gmpo-geom-mean` / 787 `rl-ktae-token-advantage`: those are GRPO-family accounts on the advantage/loss side; this problem is data construction — the expectation backprop yields role-specific SFT/DPO datasets whose products feed the DPO loss directly (the loss side is already taken and not redone).
- 784 `rl-vineppo-mc-value`: VinePPO's MC value is the mean of k per-step continuations serving as a value baseline; MALT's expectation backprop is a tree sibling mean serving as a data label, with leaf rewards from final-answer grading.
