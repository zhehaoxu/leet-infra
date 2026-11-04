"""SWEET-RL (arXiv:2503.15478, official facebookresearch/sweet_rl) --
turn-level critic ledger.

The judge injects a single ``solve``; dispatch by the ``mode`` keyword:

- mode="judge_prompt": the critic prompt with training-time information
  (kwargs: turn_input, output, ground_truth=None,
  response_template=RESPONSE_TEMPLATE); ground_truth=None means the
  no_use_ground_truth branch
- mode="judge_score": the prompt-logprobs extraction
  (kwargs: prompt_logprobs as [{token_id: (logprob, token)} | None],
  eos_token, use_sum=False) -- drops the LAST 4 positions, skips None,
  takes the logprobs AFTER the LAST eos, then sums (sum variant) or
  divides by the count (mean variant = length normalization)
- mode="rank": judge scores minus the same-shaped reference baseline,
  then descending argsort per best-of-n group
  (kwargs: judge_scores, reference_scores, best_of_n) -> reranked
  index rows
- mode="flatten": the dialogue-history flattening
  (kwargs: dialogue_history, ground_truth) -> one record per turn that
  carries "input" (input_with_ground_truth injects the prefix)
- mode="pairs": preference-pair generation
  (kwargs: data with {input, additional_outputs, judge_result},
  num_pairs=8, seed=42, no_prompt/get_sft/bestworst/eos_token) --
  ascending argsort, chosen drawn from the LOWER half, rejected from
  the UPPER half, then swapped by actual score so chosen is higher;
  no_prompt concatenates input+output+eos into the response side;
  get_sft reproduces the official chosen==repeated-first quirk
- mode="gt_pairs": the trajectory-level ground-truth preference pairs
  (kwargs: rewards flat list over k repeats, k) -> per task the argmax
  repeat is chosen and every repeat with reward strictly < max becomes
  a rejected pair (index = repeat*num_tasks + task)

All model calls (the judge RM, vLLM sampling) are explicit inputs; the
random draws are replayed with random.Random(seed) and match the
official random.seed(42) global order (verified in a sandbox).
"""
