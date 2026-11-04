# Implementation notes (787 rl-ktae-token-advantage)

## The official ledger, line by line

### compute_key_tokens.py (signal layer)

- `compute_Cohen(a, b)`: clamp both in-group rates to [0,1], then `asin√a − asin√b`.
- `fisher_exact_test(table)`: per row [a,b,c,d] (a=correct-present, b=wrong-present, c=correct-absent, d=wrong-absent), a hypergeometric exact test. Support k ∈ [max(0, R1+C1−n), min(R1, C1)], log_comb via `lgamma`, p = Σ p_k where p_k ≤ p_obs and within the support. The output vector spans 151645 entries (one per token id).
- `calculate_ig(table)`: H(Y) − H(Y|X), log2; `safe_log` logs only x>0; conditional entropy computed only when the group total >0; rows with `t_n==0` get IG 0.
- `compute_tf`: BM25 saturation. k1=2.0, b=0.5; `avg_len = (total correct tokens + total wrong tokens)/2` (**token counts**, not row counts); normalized `hist/total` then `(tf·(k1+1))/(tf+k)`.
- `get_distribution`: after `torch.unique(return_inverse=True)`, counts `torch.unique(current_rows)` per token — **distinct rows**.
- `compute_key_tokens`: empty correct or wrong pool → all-zero early return; the contingency table iterates `tokenid_set`, skipping `cc+wc==0`; fusion `((α·fmt_fisher + β·ig)·x)/(α+β)` (with γ_tf, x += tf0/(tf1+1e-5) − tf1/(tf0+1e-5)); `(sigmoid−0.5)·2·top`; the **hardcoded `result[151643] = 0.`**.
- `get_key_tokens`: `rewards > 0` selects correct, `≤ 0` wrong.

### core_algos.py::compute_ktae_outcome_advantage_and_keytokens (L155-236)

- Groups by `index` (**numpy value hashing**); scores = `token_level_rewards.sum(-1)`.
- Single-row group: mean=0, std=1, zero weight table `torch.zeros([responses.max().item()])` — **no +1**.
- Multi-row group: `torch.std` (unbiased) plus `ComputeKeyTokens(α=β=γ=1, top=1, max_token_num=responses.max().item())`.
- `scores = (s − mean)/(std + 1e-6)`; `format_weights[i] = id2key_token[index[i]][responses[i]]` (indexed by **token id**).
- Final `scores.unsqueeze(-1).tile([1, response_length]) * eos_mask + all_weight * eos_mask` — an **additive** fusion.
- `average_weight_by_mask_tensor` is dead code (defined, never called).

## Native official bugs (pinned as test cases)

1. **Small-vocab IndexError**: when max token id + 1 < 151644 the hardcoded line goes out of bounds.
2. **Single-row group + global max token id IndexError**: the single-group zero table lacks the +1, so any row containing responses.max() overflows it.

## Sandbox engineering

- /tmp/ktae_check.py (llm env torch 2.10): stubs `transformers` (the official import chain never touches a real model) and pre-registers `sys.modules['verl...compute_key_tokens']` before exec-ing the official file; core_algos contributes only the ktae function slice (cut before `compute_rloo_outcome_advantage`).
- **Fixture semantics**: Qwen-range ids (151644 included so the table reaches 151644); `index` must be numpy (tensor keys hash by object identity and KeyError).
- Long-vector EXPECTED values are stored sparsely (len + nonzero dict) with rounded comparisons (tolerating float32 lgamma/sigmoid low-bit drift across torch versions).
- **Lesson**: EXPECTED must be generated programmatically — hand-copying nonzero entries of a 151645-long vector will be wrong (the first check_fisher/ig/tf/advantage runs all failed on hand-made numbers).

## Judging protocol

The judge injects a single `solve`; `mode` kwarg dispatches eight entries. `check_*(solve)` assertion set, 12 cases.
