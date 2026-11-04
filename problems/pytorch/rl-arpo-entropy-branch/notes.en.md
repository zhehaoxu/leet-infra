# ARPO Solution Notes

## Key official details (pinned by sandbox runs)

1. **The entropy is the −Σ p·l of a top-10 truncated distribution**:
   `self.logprobs = 10` fixed, `_calc_entropy` sums once over the flattened
   top-10 logprobs — not full-vocab entropy, not per-token entropy averaged.
   Normalized by `math.log(vocab_size)`.
2. **20-token window + missing fallback**: `min(20, len(tokens))` fixed; a
   token whose logprobs cannot be read falls back to the **last** one
   (`except: logprobs[-1]`) — a vLLM first-token edge quirk.
3. **initial_entropy_dict freezes**: written only when `out_idx not in
   initial_entropy_dict` — the branch gate's delta is drift from the first
   observation, accumulating across rounds.
4. **The prob direction**: `prob = random.random() − entropy_weight × delta`,
   clamped to [0,1]; `prob > branch_probability → continue`. Rising entropy
   (delta>0) lowers prob → branches more easily; falling entropy raises it.
   rand() is the outer random source, delta the deterministic modulation —
   repeated branch attempts of the same rollout share the same delta.
5. **Three caps interlock**: remaining_slots (num_samples − rps) caps the
   total; branches_per_idx = min(beam_size − 1, remaining_slots − created)
   caps per idx; break at branches_created == remaining_slots. With
   beam_size=1, min(0, …)=0 means no branch ever.
6. **Refill: 1 branch per round + cold restart**: inactive samples restart
   from the init_inputs template of `sample_to_indices[orig][0]` (curr=init,
   masks empty, counter 0) — not a continuation of an active trajectory.
7. **Tool quota appends EOS**: `curr_inputs[out_idx].append(eos_token_id)` +
   result_masks gets a 1 + the call_limit_reached counter — requests are not
   dropped.
8. **Reorg's duplicate-to-fill**: `while len(selected) < num_samples: append(
   selected[-1])` — the floor of the branching budget; response sliced from
   prompt_len.
9. **Deviation from the paper**: the paper speaks of "the posterior entropy
   of environment-feedback tokens"; the official implementation is a top-10
   approximation with a fixed 20-token window and drift from the first-
   seen entropy — the code is authoritative.

## Boundary with neighboring problems

- `to-grpo-advantage`: GRPO's group normalization lives on the advantage
  side; ARPO's budget allocation lives on the rollout side (sample counts
  tilt toward high-entropy branches) — this problem covers only the
  rollout-side ledger.
- `to-dapo-loss`: DAPO's dynamic sampling re-samples all-correct/all-wrong
  groups within a batch; ARPO forks trajectories within one prompt during
  generation.

## Sandbox lessons

- The initial-entropy record sits **after** `current_entropy_dict[out_idx] =
  entropy` — the end anchor must be the `self.initial_entropy_dict[out_idx] =
  entropy` line, not the current-assignment line (which would cut the frozen
  semantics).
- Keep the `entropy_norm_factor = math.log(vocab_size)` line itself (the
  normalization factor); replace the get_vocab call above it.
- json dump turns tuples into lists and int keys into strings — normalize on
  the test side via `[list(b) for b in branches]` and
  `{str(k): v for k, v in rps.items()}`.
- The round triple likewise needs list() normalization.
