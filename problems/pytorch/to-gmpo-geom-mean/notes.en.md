# Solution Notes

## Approach

The three ledgers of the official callsys/GMPO `train_zero_math_gmpo.py`:

1. **compute_monte_carlo_advantages** (L407-421): token-level rewards -> trajectory advantages with three normalizations — base/drgrpo plain deviations, grpo within-group std (unbiased) + 1e-8;
2. **learning_step's six pg-loss branches** (L673-831): the gmpo family's "sign flip + conservative clipping + token geometric mean" (gmpo / noclip fixed at 1000 / seqclip sums before clamping / without_norm skips the mask.sum() division) plus the grpo family's standard PPO-clip (wider's hardcoded 0.67/1.49, else's 1±ε token-level clip + masked aggregator + loss_masks gating);
3. **KL token reward assembly** (L594-605): -coef·(logps-ref)·mask plus eos injection at the largest masked index per row (all-False rows get -1, a negative index).

Out of judging scope: vLLM actor logps collection, tokenizer/template (_apply_template), critic value learning (the ppo branch's value_pred/GAE), strategy.backward/optimizer scheduling, entropy stats, kl3 (the K3 estimate inside the beta-gated reg_loss segment), and the reinforce_update stub (assigns pg_loss_max but never sets loss — dead official code).

Boundary with existing problems:
- **to-dapo-loss / to-gspo-loss / to-cispo-k3-loss** (428/429): DAPO's token-level clip and dynamic sampling, GSPO's sequence-level ratio, CISPO's detached clipped weights; this problem is GMPO's **geometric-mean ratio + sign-flipped conservative clipping** — clipping acts on the *flipped diff* rather than on the ratio, and the loss stays a vector;
- **to-dpo-loss-backward**: analytic backward of the preference loss; this problem is the on-policy policy-gradient forward ledger;
- **the cot batch (778 etc.)**: CoT reasoning-system ledgers; this problem returns to the training-objectives side.

## Key implementation points (pinned to the official source)

- **MCA group split**: `view(-1, num_samples)` — sequences are laid out group-contiguously (num_samples per group); `repeat_interleave` (not `repeat`) maps group statistics back to every token; the grpo divisor is `(std + 1e-8)` (1e-8 inside the denominator);
- **The sign-flip three steps**: flip -> clamp -> `max(sgn·diff, clamp)` -> flip back. The max is **conservative clipping**: with adv>0 (sgn=-1), a negative diff shrunk by the clamp keeps its original value (loss not amplified); equivalently, the geometric mean of token log ratios is confined to a one-sided relaxed domain of at most ±ε;
- **The geometric mean's implementation is the ledger itself**: `exp(diff_max[mask].sum() / mask.sum())` — boolean-index the valid tokens (flattened) then sum; the divisor is `mb_response_masks.sum()` (valid token count); without_norm drops that division, seqclip moves the sum into the boolean index (scalarized, clamp acts on the sequence-total diff);
- **noclip's cliprange=1000 is the first line of the branch body** — it never reads self.args.cliprange, so the input is ignored;
- **The grpo else loss pipeline**: elementwise max -> masked_aggregator(axis=1) -> `(pg_loss * mb_loss_masks).mean()`. The mean variant divides by mask.sum() (token normalization), the sum variant by constant_normalizer (official drgrpo passes generate_max_length, length normalization); loss_masks are sequence-level 0/1 weights;
- **The three stats keys**: amax/amin are global extremes of `logprobs_diff.detach() * mb_response_masks` (**including padding positions** — multiplying by 0 makes padded diffs 0, which competes for the min; with positive diffs present the min is always ≤ 0); the zero count watches the pg_losses vector in the gmpo family and pg_loss_max in the grpo family;
- **KL injection's negative index**: `masked_indices.max(dim=1).values` gives -1 for an all-False row, and `rewards[arange, -1] += fin` is a PyTorch negative-index write — it lands on the last position rather than raising; `final_rewards.squeeze()` flattens [batch,1] to [batch] to align with row indexing;
- **batch>1 is the official native RuntimeError**: the gmpo family bools a multi-element tensor in `if mb_advantage >= 0`, the grpo family fails broadcasting `-adv[2] * ratio[2,5]` — the validation layer does not intercept, it propagates.

## Test design (13 cases)

advantage 3 (R1 with three critic_types, R2's two-group num_samples=4 group split/ordering, 5 validation ValueErrors), pg 7 (gmpo's sgn paths for positive/negative advantage + stats, the adv=0 >= boundary [-0.0], the four-variant spread including noclip's input-ignored corroboration and seqclip's amax==amin, the grpo family's four paths including wider's hardcoded clamp / sum aggregation / loss_masks=0 gating, BIG long-sequence four distinct values + seqclip scalar stats, batch=2 RuntimeError for both families, 8 validation ValueErrors), kl_reward 3 (basic ledger, all-False row -1 negative-index injection, 3 validation ValueErrors). All expected values were pinned by the official-source exec sandbox (find-boundary extraction of MCA + the six branch bodies, `self.args.*` parameterized replacements, torch 2.10.0 float64); the 24 fixtures cover the official batch=1 shape, the BIG sequence, the adv=0 boundary, and all-False mask rows.

## Deviations from the source

- The official MCA is a method with an unused `del response_masks`; this problem drops response_masks and makes critic_type/num_samples explicit inputs.
- The official branches inline the three stats appends (gmpo family `(pg_losses == 0)`, grpo family `(pg_loss_max == 0)`); this problem hoists a `_stats_record(stats, diff, mask, pg_loss, zero_target)` helper, with per-branch targets matched one by one.
- The official masked_aggregator is a class attribute (a lambda wrapping masked_sum with constant_normalizer=generate_max_length under drgrpo); this problem builds an equivalent closure at the solve layer from aggregator/constant_normalizer.
- The official stats/infos accumulate across the mini-batch loop; a single call here appends one entry each (sandbox isomorphic).
- The `reinforce_update` branch (L673-674) officially only assigns pg_loss_max without setting loss or stats — incomplete dead code, excluded from the problem.
