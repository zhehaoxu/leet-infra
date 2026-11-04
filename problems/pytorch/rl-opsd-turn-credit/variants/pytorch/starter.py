"""AgentOPSD (arXiv:2608.05987, official ZethWang/AgentOPSD; replicated by
datawhalechina/agentic-rl-lab experiments/agentopsd) -- turn-level credit
assignment ledger.

The judge injects a single ``solve``; dispatch by the ``mode`` keyword:

- mode="group": GRPO sequence advantages + initial belief
  (kwargs: rewards, epsilon) -> (advantages, mean_reward)
- mode="reshape": belief-recursion turn reshaping
  (kwargs: sequence_advantage, initial_belief, evidences,
  gamma, reshape_lambda, weight_bound, epsilon)
  -> (clipped_initial_belief, per-turn dicts with turn_index, evidence,
  cumulative_evidence, belief_before, belief_after, delta_belief,
  outcome_credit, normalized_credit, weight, advantage)
- mode="assign": whole-group assignment + degenerate-group stats
  (kwargs: trajectories as (group_id, reward, turn_evidences, n_steps)
  tuples, config params) -> (per-traj results, groups, degenerate_groups)
- mode="score": teacher-student evidence aggregation over turn token
  ranges (kwargs: steps as (action_start, action_end, logprobs) tuples,
  teacher_logprobs, student_initial_length, teacher_initial_length)
  -> (evidences, action_tokens, teacher_slices)
- mode="ppo": action-token advantage broadcasting + right-shift alignment
  (kwargs: full_tokens, steps as (action_start, logprobs) tuples,
  turn_advantages) -> dict with input_tokens, target_tokens,
  old_logprobs, advantages, action_tokens

Key semantics pinned by the official code: the initial belief is the
**unclipped group mean reward** (clip happens inside reshape at
[epsilon, 1-epsilon]); the evidence recursion c_k = gamma*c_{k-1} + e_k
starts from c_0 = 0; the outcome credit is sign(A_seq) * (B_k - B_{k-1})
-- the marginal change of the accumulated belief, not the raw gap; the
weight is clipped to [1-b, 1+b] and mixed as (1-lambda) + lambda*w; every
zip(steps, turns) fan-out truncates on the shorter side.
"""
