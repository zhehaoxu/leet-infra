# AgentOPSD Implementation Notes (790)

Source: `datawhalechina/agentic-rl-lab` `experiments/agentopsd/` (advantages.py, 196 lines of pure numpy + teacher.py::_score_turns + loss.py::build_agentopsd_ppo_datum). Paper arXiv:2608.05987. The official SkillBank is pinned to `ZethWang/AgentOPSD@0c478b2d` (comment in skills.py); this problem does not need it — the skill text is already absorbed into the teacher prefix-length difference.

## Per-entry semantics

### group_sequence_advantages (official advantages.py::60-73)

- `mean_reward = sum(rewards)/len(rewards)`, `std = np.std(rewards, ddof=0)`, `adv = (r−mean)/(std+ε)`.
- **Returns (advantages, mean_reward)** — mean_reward is the unclipped group mean, passed to reshape as the initial belief. ε is caller-supplied (assign passes config.epsilon).
- All-equal groups: std=0 → all-zero advantages; single-trajectory groups likewise. These two degenerate shapes are the source of the degenerate check and the ppo filter.

### _sigmoid (official 76-83)

Numerically stable two-branch: `value >= 0` takes `1/(1+exp(−v))`, otherwise `exp(v)/(1+exp(v))`. Both branches agree at v=0.

### reshape_turn_advantages (official 86-156, the core)

Field-by-field mapping onto the TurnCredit dataclass, order pinned:

1. `b0 = min(max(initial_belief, ε), 1−ε)` (**initial_belief is the unclipped group mean**); `logit_b0 = log(b0/(1−b0))`.
2. Recursion (c_0=0): `cumulative = γ·cumulative + evidence`; `belief_after = σ(logit_b0 + cumulative)`; `raw.append((evidence, cumulative, belief_before, belief_after))`; `belief_before = belief_after` advances as a chain.
3. `sign = +1 / −1 / 0` (by the sign of sequence_advantage, with an explicit zero branch).
4. `outcome_credit_k = sign · (after − before)` — **the before/after from raw**, not belief_after vs b0.
5. `z = (credit − mean(credits)) / (std(credits, ddof=0) + ε)`; with a single turn std=0 and credit itself=0 → z=0.
6. `weight = min(max(1 + b·z, 1−b), 1+b)` (default [0.8, 1.2]); `multiplier = (1−λ) + λ·weight` (default [0.9, 1.1]); `A_turn = A_seq · multiplier`.
7. turn_index starts at 1; `delta_belief = after − before` (differs from outcome_credit by the sign factor).

**Zero-advantage trajectories**: sign=0 → all-zero credits (`0.0` and `−0.0` both appear; tests use tolerance) → all-zero z → all-one weights → multiplier=1 → all-zero A_turn. The evidence/belief ledger is still computed.

**Clip boundaries**: initial_belief=1.0 → b0=0.9999, with evidence=0 the after/before differ by 1e-16-level float noise; initial_belief=0.0 → b0=1e-4, σ(logit(1e-4)) round-trips to 9.99999e-05 and −1.35e-20 noise. Tests all use a 1e-9 tolerance.

### _score_turns (official teacher.py::63-87)

- `prefix_offset = teacher_initial_length − student_initial_length`. The teacher prefix (with the skill prompt) is longer; **every action range shifts by +offset** (teacher_start = action_start + offset).
- Per turn: `numeric = [float(v) for v in teacher_logprobs[start:end]]`; `evidence = sum(t − s for t, s in zip(numeric, student_logprobs))`.
- **zip truncation**: when the teacher slice and the student logprobs differ in length, sum on the shorter side. Real official data keeps them equal (the action range is exactly the student-generated tokens), so truncation is implicit behavior — the fixture pins it with (action_start=6, action_end=10, 2 logprobs).
- action_tokens accumulates `len(numeric)` (the teacher-side length, not the post-zip length).
- The official code also writes numeric/float(evidence) back to step.teacher_logprobs/step.teacher_evidence — this problem replaces that with the returned (evidences, action_tokens, slices).

### assign_agentopsd_advantages (official 159-196)

- `groups: dict[str, list]`, keyed by `str(group_id)` (officially a defaultdict; same semantics as setdefault).
- Each group computes sequence advantages and the initial belief first; **the degenerate check runs before the write-back**: `all(abs(adv) <= ε)`.
- Each trajectory then runs `reshape_turn_advantages(adv, initial_belief, turn_evidences)` and writes back: `step.teacher_evidence = turn.evidence`, `step.turn_advantage = turn.advantage`, truncated by `zip(trajectory.steps, credit.turns)` — **fewer steps than evidences fills only the prefix**.
- Returns AdvantageStats(groups, degenerate_groups). This problem returns (results, groups, degenerate_groups) with per-traj dicts (group_id/sequence_advantage/initial_belief/turn_advantages/step_evidences/step_turn_advantages); **results keep input order** (the official attribute write-back preserves order naturally; this implementation maps by id — list.index() would misalign two trajectories with identical evidences).

### build_ppo_arrays (official loss.py::42-77)

- Zero-fill two arrays over all tokens first (old_logprobs_by_token / advantages_by_token).
- `zip(steps, turn_credits)`: position = `action_start + offset` (**no action_end needed**; official loss.py only uses action_start + enumerate(logprobs)); broadcast the old_logprob and that turn's credit.advantage in; action_tokens accumulates len(logprobs).
- Right-shift alignment: `input_tokens = full[:-1]`, `target_tokens = full[1:]`, `old_logprobs = old_by_token[1:]`, `advantages = adv_by_token[1:]` — the label at position p is token p+1. **The last action token's logprob lands at the tail of [:-1] and is not dropped** (advantages_by_token is as long as the full token list; after [1:] it still covers).
- zip truncation: when steps outnumber turn_advantages the extra steps write nothing.
- The official `build_training_datums` filters `sequence_advantage != 0.0` — a degenerate group is excluded from training wholesale. The tests pin this semantics with a Python expression inside the assign case.

## Sandbox and tests

- Sandbox script `/tmp/opsd_check.py`: imports the official advantages.py directly via sys.path (pure numpy, zero deps) + a character-faithful `_score_turns` replica + a ppo-array replica; dumps `/tmp/opsd_expected.json` and `/tmp/opsd_expected_cfg.json` for the custom config (γ=0.5, λ=0.8, b=0.5, ε=0.01). EXPECTED is fully program-generated into test_cases.py (16 groups).
- Cross-check: all five solve entries × all fixtures match the dump field-by-field within a 1e-9 tolerance.
- Coverage matrix: three GRPO groups (normal / all-equal degenerate / single) · positive/negative/zero-advantage trajectories with full fields · both clip bounds · custom config with four turns plus a single turn · prefix_offset over three turns with a manual slice check · zip truncation on both the score and ppo sides · assign grouping/degenerate stats/truncated write-back/training filter · ppo right-shift alignment and zero-fill.

## Mutation checklist (mut790 pending)

See the 7 check functions in test_cases.py; easy-to-miss spots: ddof=0 vs ddof=1, the min/max order inside clip, whether c_0 is 0 (first-turn cumulative), the sign(0) branch, the weight clip bounds, the (1−λ) base of the multiplier, the prefix_offset direction, zip truncation, the [1:] right shift, and <= vs < in the degenerate check.

## Boundaries

- `to-grpo-advantage`: that problem is the group standardization itself on the advantage side; this one adds three reshaping layers after standardization (belief recursion + marginal credit + bounded multiplier) and drops the credit granularity from sequence to turn.
- 784 `rl-vineppo-mc-value`: VinePPO replaces the value net with MC return estimates (k continuations per step); this problem samples nothing — one teacher forward and a logprob-difference ledger.
- 786 `rl-arpo-entropy-branch`: ARPO branches during rollout driven by entropy; this problem reshapes advantages after rollout.
