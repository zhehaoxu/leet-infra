# notes — inf-nsa-block-dispatch (order 683)

## Source

- Paper arXiv 2502.11089 (PDF downloaded, extracted via workbuddy-pdf, §3.3 verified verbatim). **No official implementation exists**: the deepseek-ai/NSA repo is 404 (API confirmed Not Found); community implementations (tilde-research/nsa-impl, qflen/nsa-from-scratch, dhcode-cpp/NSA-pytorch, FLA's parallel kernel) disagree on details, so the paper's Eqs. (5)-(12) are the sole authority.
- Formula ledger (verbatim from the paper):
  - (7) Compression: K̃^cmp_t = {φ(k_{i·d+1:i·d+l}) : 0 ≤ i ≤ ⌊(t−l)/d⌋}, giving ⌊(t−l)/d⌋+1 compressed keys. Blocks overlap when d < l. φ is an MLP with intra-block position encoding.
  - (8) p^cmp_t = softmax(q_t^T K̃^cmp_t) ∈ R^{⌊(t−l)/d⌋+1}.
  - (9) Cross-granularity: p^slc_t[j] = Σ_{m=0}^{l'/d−1} Σ_{n=0}^{l/d−1} p^cmp_t[(l'/d)·j − m − n]. **Degenerates to identity when l=l'=d** (the paper explicitly says "directly obtain … straightforwardly"). The PDF extraction makes the formula's index base (0/1) unreliable, so the problem avoids mixed granularity and uses only the l=d=l' mainline.
  - (10) GQA group sharing: p^slc,t' = Σ_{h=1}^{H} p^slc,(h)_t — a **sum**. Motivation: decoding-time KV loading equals the union of all heads' selections in the group, so group-consistent selection minimizes loading.
  - (11) I_t = {i | rank(p^slc_t'[i]) ≤ n}, rank descending with rank=1 highest. **The tie-break is unspecified in the paper** → the problem pins "ties rank by ascending block index (stable)".
  - (12) K̃^slc_t = Cat{k_{i·l'+1:(i+1)·l'} : i ∈ I_t} — whole-block concatenation.
  - (6) N_t = Σ_c size[K̃^c_t], with the paper requiring N_t ≪ t.
  - Sliding window: K̃^win_t = k_{t−w:t} (includes the current token).
  - Gates: g^c_t from an MLP + sigmoid over input features; Eq. (5) output = Σ_c g^c_t · o^c_t. Independent KV per branch is the paper's anti-shortcut design (§3.3.3).

## Abstractions

- No softmax/matmul (not hand-computable; the attention body is covered by other problems): q/k/v never appear — p_cmp is given directly as "already-softmaxed compressed-block scores"; branch outputs o are given directly.
- Causal convention: L tokens generated (0..L-1), complete compression blocks = L//l ((L−l)//d+1 with d=l equals L//l ✓ hand-checked with L=5, l=2 → blocks [0,2),[2,4)).
- Selection candidates = complete blocks ((i+1)·l ≤ L). Mixed-granularity cases (l ≠ l') are not tested (Eq. 9's index base is dubious, see above).
- Errors raise ValueError (repo convention).

## Ledger points (nailed by tests)

1. **Group sum vs mean/max** (check_gqa_group_sum; mutations gqa-mean / shared-first-only) — constructed so a single head's ranking differs from the group sum's, pinning the Quest-style "per-head independent selection" mistake.
2. **Descending rank + stable ties take the lower index** (check_tie_stable_low_index; mutations rank-asc / tie-large).
3. **top_n > block count selects all, but slc_len still counts actual blocks** (check_top_n_larger_than_blocks; mutation slc-len-topn catches slc_len = top_n·l).
4. **selected_blocks output ascending** (rank order ≠ ascending construction in check_selected_sorted_ascending; mutation sel-keep-rank-order).
5. **Sliding window min(window, L)** (check_window_cap / check_window_normal; mutations win-uncap / win-off).
6. **Linear gated aggregation + silent zero-gate branches** (check_gated_aggregation / check_zero_gate_branch_silent; mutations gate-drop-win / gate-add).
7. **n_blocks = L//l causal ledger** (check_l_equals_one_degenerate's l=1 degenerate; mutation cmp-len-plus catches cmp_len).
8. Input validation (check_invalid_inputs, 7 invalid shapes).

## Mutation record

11 observable mutations all CAUGHT: `gqa-mean`, `rank-asc` (selects lowest), `tie-large` (ties take higher index), `sel-keep-rank-order` (output in rank order), `slc-len-topn` (slc_len from top_n instead of actual blocks), `win-uncap`, `win-off` (min(w,L+1)), `gate-drop-win`, `gate-add` (g·a → a), `cmp-len-plus`, `rank-strict` (selects one fewer), `shared-first-only`.

**Equivalent-mutation analysis (sel-no-clip)**: `order[: min(top_n, n_blocks)]` → `order[: top_n]` is equivalent — Python slicing truncates naturally, the selection set is unaffected; the real ledger point is the min in slc_len (already covered by slc-len-topn).

## Hand-computed anchor pitfalls

- The oracle's sort key was written backwards ((shared[i], -i) ascending = selects the lowest) — the cross-check FAIL direction exposed it exactly: "expected [0,2] got [1,3]"; the solution was right and the oracle was fixed.
- The first draft of check_window_cap left an invalid L=3,l=4 call (n_blocks=0 conflicts with head length 1) → removed.
- After renaming _compare's keyword parameters (window/branch_outputs), all call sites were updated.

## Boundaries with adjacent problems

- `sliding-window-attention` (operators): the pure causal sliding-window attention body.
- kvcache problems: KV eviction/recycling; NSA is **training-aware** block selection (gradients flow through the gating; selection is discrete but uses no k-means/SimHash).
- This problem: the three-branch **dispatch ledger** (GQA sharing, top-n, branch lengths, gating) — no attention matmul/softmax body.
