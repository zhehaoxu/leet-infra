# GMPO: Token Geometric-Mean Advantages and Sign-Flip Conservative Clipping (MC Advantage Three-Way + Six pg-loss Branches + KL Token Reward Injection)

## Problem

GMPO (arXiv:2507.20673, official repo `callsys/GMPO`) replaces GRPO's within-group arithmetic-mean advantage with a **geometric mean** over token-level log ratios (sum in log space, then exponentiate), plus sign-flipped conservative clipping. This problem takes the three ledgers from `train_zero_math_gmpo.py`, all as explicit inputs:

```text
mode="advantage": rewards(token-level 2-D) + num_samples + critic_type(None|"grpo"|"drgrpo") -> advantage list
mode="pg":        variant + new_logps + old_logps + response_masks + advantage + loss_masks
                  (+cliprange + aggregator + constant_normalizer) -> {"loss", "stats"}
mode="kl_reward": logps + ref_logps + response_masks + final_rewards + coef -> {"rewards", "eos_indices"}
```

## Ledger rules (pinned to the official source)

`mode="advantage"` (official `compute_monte_carlo_advantages` L407-421):

```text
rewards = rewards.sum(-1)                      # fold token rewards to trajectory level
values = rewards.view(-1, num_samples).mean(1) # within-group arithmetic mean
values = values.repeat_interleave(num_samples) # back to every token
advantages = rewards - values
critic_type="grpo": divide by group std (torch default: unbiased, n-1 denominator) + 1e-8
critic_type="drgrpo"/None: no normalization; advantages returned as-is
```

`mode="pg"` (official `learning_step`, six branches L673-831). The four gmpo-family variants share the "sign-flip + conservative clipping" skeleton, differing only in how the diff is folded and normalized:

```text
gmpo:
  sgn = -1 if advantage >= 0 else 1        # >= boundary goes to -1
  diff = new - old; flip -> clamp(-ε, +ε) -> max(flip(diff), clamp) -> flip back
  ratio = exp( diff_max[mask].sum() / mask.sum() )   # token geometric mean
  pg_losses = -advantage * ratio            # vector; loss gets no masked mean
gmpo_noclip:  same as gmpo, cliprange fixed at 1000 internally (input ignored)
gmpo_seqclip: diff is first summed over the mask into a scalar, then
              flip-clamp-max; ratio = exp(diff_max / mask.sum()); stats amax == amin == sum
gmpo_without_norm: ratio = exp( diff_max[mask].sum() ), no division by mask.sum()
```

The two grpo-family branches use the standard PPO-clip skeleton with different clamp domains:

```text
grpo_clip_wider: hardcoded clamp(0.67, 1.49) (the input cliprange does not participate)
grpo (else fallback):
  ratio = exp(diff) per token
  pg_losses2 = -adv * clamp(ratio, 1-ε, 1+ε)
  pg_loss_max = max(pg_losses, pg_losses2) elementwise
  pg_loss = masked_aggregator(pg_loss_max, mask, axis=1)
            (aggregator="mean" divides by mask.sum(); "sum" by constant_normalizer)
  loss = (pg_loss * loss_masks).mean()      # scalar
```

The three stats keys each get one append per mini-batch: `logprobs_diff_max` /
`logprobs_diff_min` are the global amax/amin of `diff * mask` (seqclip's diff is a
scalar, so amax == amin); `zero_pg_loss_count` is `(pg_loss_max == 0).sum()`
(for the gmpo family the target is the pg_losses vector).

Official quirks (pinned by tests): advantage is a tensor, so `sgn = -1 if adv >= 0 else 1` raises RuntimeError from booling a multi-element tensor at batch>1; and `-adv[batch] * ratio[batch, seq]` fails to broadcast in the grpo family — **both pg families only run with one sequence per mini-batch**.

`mode="kl_reward"` (official L594-605):

```text
kl_rewards = -coef * (logps - ref_logps) * mask
eos_indices = largest masked index per row (-1 for an all-False row)
rewards[row, eos_indices] += final_rewards.squeeze()
```

The -1 for an all-False row is a **negative index**: the final reward is injected into that row's **last position** (PyTorch negative-index semantics), not an error.

## Key quirks (official semantics)

- The grpo normalization in `advantage` uses `std(dim=1)` **unbiased by default** (n-1 denominator), unlike Dr.GRPO's n denominator; `+1e-8` sits in the denominator;
- gmpo flips the sign **before** the clamp and restores it **after** the max: `max(sgn·diff, clamp(sgn·diff))` is "conservative clipping" — the positive direction is kept in full, the negative direction only keeps what the clamp enlarged;
- The geometric-mean ratio sums in **log space** before exp; the divisor is `mask.sum()` (valid token count), not the sequence length;
- gmpo-family loss is a **vector** (one ratio per sequence) with no masked mean — unlike the scalar path of the grpo family;
- noclip's ε=1000 is a **branch-internal constant**, not read from args.cliprange;
- seqclip's stats amax == amin: once the diff folds into a scalar, `amax(scalar*mask)` equals `amin`;
- The batch>1 RuntimeError is the official native behavior (tensor booling / broadcast failure), not validation added by this problem.

## Input conventions

Tensors are constructed as float64; `response_masks` holds 0/1 (converted to bool internally); `final_rewards` is `[batch, 1]` (squeezed internally); `aggregator="sum"` requires a positive `constant_normalizer` (official drgrpo uses generate_max_length).
