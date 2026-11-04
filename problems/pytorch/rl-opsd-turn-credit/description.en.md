# AgentOPSD: Turn-Level Credit Ledger (790)

**Source**: datawhalechina/agentic-rl-lab `experiments/agentopsd` (adapted from ZethWang/AgentOPSD, arXiv:2608.05987, Tsinghua/Zhejiang/Meituan).

## Problem

GRPO broadcasts the sequence-level advantage to every action token in a trajectory — in long-horizon ALFWorld rollouts the early detours and the final key action get encouraged together. AgentOPSD adds a privileged self-distillation layer: a Self-Teacher (the same parameter snapshot plus access to the SkillBank) re-scores **the same trajectory the Student executed**, token-level logprob gaps are summed along turns (a stretch of actions between two environment observations) into evidence; but **an isolated local gap is not the credit directly** — the evidence is recursed into a Bayesian-inspired accumulated belief, the credit is the belief's marginal change ΔB, and the sequence advantage is reshaped by a bounded multiplier. pytrio/Trajectory dependencies are fully parameterized: per-step logprobs, action token ranges, and final rewards are all explicit inputs; the whole ledger is deterministic.

## What to solve

`solve(mode=..., ...)` five entries (mode kwarg dispatch, numpy + stdlib):

1. `mode="group"` (rewards, epsilon) → (GRPO sequence advantages, the unclipped group mean reward as initial belief)
2. `mode="reshape"` (sequence_advantage, initial_belief, evidences[, γ/λ/b/ε]) → (clipped initial belief, ten-field per-turn ledger)
3. `mode="assign"` (trajectories: (group_id, reward, turn_evidences, n_steps) tuples[, config]) → (per-traj results, group count, degenerate-group count)
4. `mode="score"` (steps: (action_start, action_end, logprobs) tuples, teacher_logprobs, student/teacher prefix lengths) → (per-turn evidence, action token count, teacher slices)
5. `mode="ppo"` (full_tokens, steps: (action_start, logprobs) tuples, turn_advantages) → {input_tokens, target_tokens, old_logprobs, advantages, action_tokens}

## Key official semantics (pinned from source, see notes.md)

- The initial belief is the **unclipped group mean reward**; the clip happens only inside reshape at [ε, 1−ε]
- The evidence recursion c_k = γ·c_{k-1} + e_k starts from c_0 = 0 and belief_before advances as a chain (not reset per turn)
- **outcome_credit = sign(A_seq)·(B_k − B_{k-1})**: sign(0)=0 → zero-advantage trajectories get all-zero credits and all-one weights
- The weight is a three-stage account: z-standardize → w = clip(1+b·z, 1−b, 1+b) → multiplier = (1−λ)+λ·w; **magnitude redistribution only, never direction flipping**
- score: prefix_offset = teacher prefix length − student prefix length shifts every action range; zip truncation semantics
- ppo: non-action tokens are zero-filled; right-shift alignment (input=tokens[:-1], target=tokens[1:], logprobs/advantages from [1:])
- Trajectories with sequence_advantage == 0 (degenerate groups) are excluded from training

## Scoring

There is no "score" here — judging is ledger comparison: every returned field maps one-to-one onto the official advantages.py / teacher.py / loss.py, and the test EXPECTED values were dumped from the official source in a sandbox.

## Judging environment

torch + numpy, no pytrio / transformers; no randomness (fully deterministic ledger).
