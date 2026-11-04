# GRESO: Pre-Rollout Selective Filtering (788)

**Source**: arXiv:2506.02177 (Act Only When It Pays / GRESO), official `Infini-AI-Lab/GRESO` (verl fork).

## Problem summary

Historical reward traces predict each prompt's rollout value: prompts that keep answering all-correct (too easy) or all-wrong (unsolvable) are skipped **before rollout**; the saved budget is consumed by the zero-variance-filtered adaptive refill; the easy/hard filtering probabilities adapt across steps against measured ratios. Fully deterministic: traces, batches, and random sources are all explicit inputs.

## What to solve

`solve(mode=..., ...)` with seven entries (mode kwarg dispatch, stdlib only):

1. `mode="id"` (data_sources, indices) → `["{source}_{index}", ...]`
2. `mode="trace"` (ops, data_id) → append-only (epoch, reward) trail
3. `mode="skip"` (ops, current_epoch, data_id, variant, ...) → single-example skip decision (five variants)
4. `mode="filter"` (ops, current_epoch, batch, p_easy, p_hard, rand_values) → (selected, skip_log, counts)
5. `mode="refill"` (reminder, zero_variance_ratio, n, default_bs, ...) → (bs, last_sampling, used_full)
6. `mode="adapt"` (epoch, hard_ratio, easy_ratio, p_hard, p_easy, ...) → (p_hard, p_easy)
7. `mode="zero_variance"` (rewards, n) → (group flags, kept flat indices)

## Key official semantics (pinned against the source, see notes.en.md)

- easy = trailing run of rewards ≥0.98; hard = latest reward ≤0.11 (the solvable-example gate comes first) plus a trailing run ≤0.11
- backoff = randint(0, k·count) anchored at the **whole-trace** max epoch; probabilistic skip_prob = 1 − max(p^count, 0.01) with no count==0 short-circuit
- **Lazy rand draws**: short-circuited checks never consume the random source
- Refill estimate = int(reminder/(1−zvr)/n·1.25), clamped to [64, default]; estimated ≤ default sets last_sampling
- p adaptation: ratio ≥ target (hard=target·2/3, easy=target/3) → p −0.01 (filter more) else +0.01, clamped to [0.05, 0.95], frozen at epoch 0
- Zero-variance filtering: a group with reward max==min is dropped whole

## Scoring

- A set of `check_*` functions; EXPECTED values dumped from the official code sandbox (/tmp/greso_check.py, random sources parameterized)
