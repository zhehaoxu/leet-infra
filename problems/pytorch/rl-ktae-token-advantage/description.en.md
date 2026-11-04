# KTAE: Token-Level Advantage Ledger (787)

**Source**: arXiv:2505.16826 (KTAE), official `ZNLP/KTAE` (verl fork).

## Problem summary

Outcome rewards score whole sequences. KTAE splits the batch into correct (reward > 0) / wrong (≤ 0) rollout pools, builds a per-token 2×2 contingency table (in-group rows containing the token / cross-group rows / respective absences), and fuses four statistical signals — Fisher exact test, information gain, BM25-saturated term frequency, arcsine-Cohen distance — into a "key token weight table". The advantage is the group-standardized outcome score (GRPO-style) broadcast over the eos mask **plus** the weight table indexed by token id. Fully deterministic: rollouts, rewards, and group indices are all explicit inputs.

## What to solve

`solve(mode=..., ...)` with eight entries (mode kwarg dispatch, torch float32):

1. `mode="distribution"` (tensor, mask) → `{token_id: distinct rows containing it}`
2. `mode="appear_table"` (responses_ids, mask, rewards) → contingency tensor of rows `[cc, wc, tc−cc, tw−wc]`
3. `mode="fisher"` (table) → formatted Fisher signal `exp(−2p)`, p==1 → 0
4. `mode="ig"` (table) → information gain vector (log2)
5. `mode="tf"` (correct, correct_mask, wrong, wrong_mask, table) → BM25-saturated frequencies [N+1, 2]
6. `mode="cohen"` (table) → arcsine−sqrt distance vector
7. `mode="key_tokens"` (responses_ids, mask, rewards[, alpha, beta_ig, gamma_tf, top, max_token_num]) → weight table
8. `mode="advantage"` (token_level_rewards, responses, eos_mask, index[, epsilon]) → [bsz, response_length]

## Key official semantics (pinned against the source, see notes.en.md)

- Table sized `max_token_num + 1` (the caller passes `responses.max().item()`), hardcoded `result[151643] = 0`
- Single-row groups allocate their zero table as `responses.max()` (**no +1** — inconsistent with multi-row groups; a native bug)
- `get_distribution` counts distinct rows (unique rows per token), not frequencies
- Fisher p-values via a lgamma numeric approximation with support-masked accumulation; p==1 squashed to 0
- Key-token sign comes from the Cohen difference: correct-only → sigmoid-saturated +1, wrong-only → −1, equal frequencies → 0
- The advantage fusion is **additive** (standardized tile + weights, each masked); an empty correct/wrong pool → all-zero weights
- `index` hashes by value (official passes numpy); `torch.std` is unbiased

## Scoring

- A set of `check_*` functions; EXPECTED values dumped from the official torch code (sandbox /tmp/ktae_check.py)
- Long vectors (~151645) stored sparsely; rounded numeric comparisons tolerate float32 low-bit drift
