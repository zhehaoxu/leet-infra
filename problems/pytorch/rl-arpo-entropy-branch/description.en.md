# ARPO: Entropy-Aware Rollout Branching

## Problem

In multi-turn tool-call agent RL, the model's distribution uncertainty spikes
right after environment feedback enters the context. ARPO (Agentic Reinforced
Policy Optimization, arXiv:2507.19849) spends the sampling budget by
**adaptively branching at high-entropy moments after environment feedback**:
each rollout monitors the drift of its top-10 truncated entropy (relative to
its first-observed initial entropy), rising-entropy trajectories branch more
easily, falling-entropy ones rarely; every sample still ends with exactly
num_samples trajectories. This problem replicates the deterministic accounting
of the official vllm_rollout_with_tools.py (vLLM engine/tool executor/logging
stripped, rollout outputs and the random source fully parameterized) via five
`solve(mode=...)` entries:

1. **entropy**: per rollout take the first min(20, len(tokens)) tokens' top-10
   logprobs flattened (missing falls back to the last), entropy = −Σ p·l
   (p=exp(l)), empty → 0, normalized by log(vocab_size); initial_entropy_dict
   written on first sight only (frozen afterwards).
2. **branch**: per sample remaining_slots = num_samples − rps (skip ≤0); per
   active idx branches_per_idx = min(beam_size − 1, remaining_slots −
   branches_created); per branch prob = clamp(rand() − weight×delta, 0, 1),
   prob > branch_probability skips.
3. **refill**: inactive samples get exactly 1 branch per round, templated
   from the first index (init_inputs cold restart).
4. **round**: tool call under quota → ("tool", c+1, []); over quota →
   ("limit_eos", c, [eos]); length below the cap → continue else stop; EOS →
   stop.
5. **reorg**: per sample the first num_samples indices, duplicating the last
   to fill shortfalls.

## I/O

- `entropy`: outputs + active_indices + vocab_size + initial_entropy_dict
  → (current, initial)
- `branch`: active_by_sample + both entropy dicts + rps + num_samples +
  beam_size + branch_probability + entropy_weight + rand_fn → (branches, rps)
- `refill`: active_by_sample + rps + num_samples + sample_to_indices
  → (refills, rps)
- `round`: 8 params → (action, new_counter, extra_tokens)
- `reorg`: batch_size + sample_to_indices + num_samples + curr_inputs +
  result_masks + prompt_lens → (sequences, masks)

## Example

Sample 0 with two active rollouts [3,4], num_samples=4, rps=2, beam_size=3:

- idx3 entropy 0.9 (init 0.5, delta +0.4): prob = rand − 0.2 → branches for
  most rand draws
- idx4 entropy 0.5 (init 0.7, delta −0.2): prob = rand + 0.1 → rarely branches
- 2 slots assigned in active order, break at the cap

## Edge cases and pitfalls

- initial_entropy_dict freezes: delta accumulates across rounds, never
  overwritten after first sight.
- Three caps interlock: with beam_size=1, min(0, …)=0 means no branch ever.
- Refill cold-restarts from init_inputs, not a continuation of an active
  trajectory.
- Hitting the tool quota appends EOS (does not drop the request).
- The entropy is a top-10 approximation with a fixed 20-token window — the
  code is authoritative over the paper's phrasing.
