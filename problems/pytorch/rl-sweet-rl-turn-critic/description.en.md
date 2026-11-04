# SWEET-RL: Turn-Level Critic Ledger (792)

**Source**: facebookresearch/sweet_rl `scripts/` (rank_best_of_n.py / generate_random_pairs_from_ranks.py / sample_best_of_n.py / evaluate_code.py), paper arXiv:2503.15478 (training lives in the companion fork YifeiZhou02/collab_openrlhf: multi-turn DPO + length normalization).

## Problem

Multi-turn agent RL's credit assignment uses a turn-level critic with access to **training-time information**: the critic's prompt injects the ground truth ("In light that the final answer is: ..."), scoring every turn's candidate outputs; the policy trains with length-normalized DPO on the step-level preference pairs the critic ranking produces. All model calls are explicit inputs — logprob lists, token lists, judge scores, and the reference baseline — leaving only the scripts' deterministic accounting.

## What to solve

`solve(mode=..., ...)` six entries (mode kwarg dispatch, numpy + stdlib):

1. `mode="judge_prompt"` (turn_input, output, ground_truth=None) → the critic prompt (GT prefix + RESPONSE_TEMPLATE tail)
2. `mode="judge_score"` (prompt_logprobs, eos_token, use_sum=False) → the score-extraction ledger
3. `mode="rank"` (judge_scores, reference_scores, best_of_n) → descending rerank indices after subtracting the reference baseline
4. `mode="flatten"` (dialogue_history, ground_truth) → per-turn records (input_with_ground_truth injection)
5. `mode="pairs"` (data, num_pairs=8, seed=42, no_prompt/get_sft/bestworst) → preference pairs
6. `mode="gt_pairs"` (rewards, k) → trajectory-level GT preference-pair indices

## Key official semantics (see notes.md)

- The judge four-step account: `[:-4]` tail drop (4 empty-response tokens) → skip None (logprob/token in sync) → everything **after the LAST eos** → sum or mean (use_sum = the length-normalization switch)
- Training-time info injected at three sites: the flattened input_with_ground_truth / the critic prompt prefix / the GT preference text prefix — **no space** between prefix and text
- rank: subtract the same-shaped reference baseline (reshape(-1, best_of_n)) then descending argsort
- The draw: ascending argsort taken **by position** — chosen from the lower half, rejected from the upper half, then swapped by actual score so chosen is higher (never the absolute best; only bestworst takes the extremes)
- The get_sft official quirk: chosen==rejected both use the first index
- gt_pairs: the argmax repeat is chosen; only rewards **strictly <** max produce pairs; index = repeat*num_tasks+task
- random.seed(42) is the global seed; the choice order and shuffle replay exactly

## Scoring

Judging is ledger comparison: the seed-42 draw sequence was cross-checked pair-by-pair against the official scripts.

## Judging environment

torch + numpy; no model calls (logprobs are inputs).
