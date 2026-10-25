# RL Systems: The Reward-Normalization Batch-Statistics Pipeline

## Problem

Raw rewards carry no scale: a reward model may output values in $[-3,7]$ or in $[10^3,10^4]$, and feeding them straight into a policy gradient makes the gradient magnitude drift with the reward model. So RLHF trainers bolt a **normalization pipeline** onto the reward output: an EMA ledger of running mean/variance, a switch choosing "batch statistics / group statistics / ledger statistics", and a switch choosing "just subtract the mean / also whiten by the standard deviation". OpenRLHF's `reward_norm` offers the `prompt/group/running_state/batch/none` settings and keeps a `RunningMoments` object; verl has equivalent reward/advantage normalization switches. The mechanism is the same everywhere — and so are the pitfalls: pick the wrong statistic source or the wrong update order and the output scale silently drifts away from 1, training neither diverges nor converges fast, and it is miserable to debug.

This problem derives no loss or advantage formula (that is `to-grpo-advantage` / `to-gae`'s job). It treats the pipeline as a **data-flow system**: given a stream of reward batches in arrival order, produce the normalized values, the ledger history, and a probe that exposes scale mismatch.

## Rules

Six inputs:

| Name | Type | Meaning |
|---|---|---|
| `reward_batches` | float $(T,B)$ | $T$ batches in arrival order, $B$ rewards each |
| `group_ids` | int64 $(B,)$ | prompt-group id of each reward inside a batch (the same ids are reused for every batch) |
| `stat_source` | 0 / 1 / 2 | statistic source: 0 = current batch global, 1 = running ledger, 2 = within-batch groups |
| `momentum` | float $\in[0,1]$ | EMA coefficient in the **PyTorch BatchNorm convention**: $r\leftarrow(1-m)r+m\,x$ |
| `whiten_std` | 0 / 1 | 1 = subtract mean then divide by std; 0 = subtract mean only |
| `eps` | float $>0$ | numerical term added to the variance before the square root |

Process batches in order, $t=0,1,\dots,T-1$:

1. Pick the mean/variance **used to normalize this batch** according to `stat_source`:
   - `0`: this batch's global mean and population variance (ddof=0);
   - `1`: the ledger values **before** this batch's update, `running_mean[t]` / `running_var[t]` (initialised to $0.0$ / $1.0$, like BatchNorm);
   - `2`: **within-batch group statistics** — rewards sharing a `group_ids` value use that group's mean and population variance.
2. $\text{centered}=x-\mu_{\text{sel}}$; if `whiten_std==1` divide by $\sqrt{\text{var}_{\text{sel}}+eps}$, otherwise keep the centred value.
3. Compute the probe `out_rms[t]` $=\sqrt{\operatorname{mean}(\text{normalized}[t]^2)}$ ($\approx 1$ when properly whitened).
4. Finally update the ledger with this batch's **global** mean and population variance: `running_mean[t+1] = (1-m)·running_mean[t] + m·batch_mean`, likewise for the variance.
   Note: **the ledger is always driven by batch-global statistics**, independently of `stat_source` (the switch decides what is used to normalize, not what gets written into the ledger).

Return a 4-tuple (all float32): `normalized` $(T,B)$, `out_rms` $(T,)$, `running_mean` $(T+1,)$, `running_var` $(T+1,)$. Index $0$ of `running_*` is the initial value $0.0/1.0$ and index $t+1$ is the state **after** batch $t$ was written — so "what was used" and "what was written" sit visible on one timeline.

## Example

`reward_batches=[[1,2,3,4],[2,3,4,5],[10,20,30,40]]`, `group_ids=[0,0,1,1]`, `momentum=0.5`, `whiten_std=1`, `eps=1e-5`:

`stat_source=0` (batch-global, the textbook choice):

```
normalized   = [[-1.341635, -0.447212,  0.447212,  1.341635],
                [-1.341635, -0.447212,  0.447212,  1.341635],
                [-1.341641, -0.447214,  0.447214,  1.341641]]
out_rms      = [0.999996, 0.999996, 1.0]          # every batch whitened to unit scale
running_mean = [0.0, 1.25, 2.375, 13.6875]        # init 0 → EMA tracking (2.5, 3.5, 25)
running_var  = [1.0, 1.125, 1.1875, 63.09375]    # init 1 → EMA tracking (1.25, 1.25, 125)
```

Same data with `stat_source=1` (normalize new data with old statistics):

```
normalized[0] = [0.999995, 1.99999, 2.999985, 3.99998]        # first batch is effectively unnormalized
normalized[2] = [6.99715, 16.17374, 25.350332, 34.526924]     # rewards jumped to 10~40, ledger variance stuck at 1.19
out_rms       = [2.738599, 2.368768, 23.15868]                # 23x scale mismatch, the probe screams
```

With `stat_source=2` (group statistics, the GRPO source choice): two rewards per group collapse to $\pm1$:

```
normalized   = [[-0.99998, 0.99998, -0.99998, 0.99998], ...]
out_rms      = [0.99998, 0.99998, 1.0]
```

With `momentum=0.0` and `stat_source=1` the ledger freezes at $(0,1)$, the output equals the raw rewards (up to a tiny $1/\sqrt{1+eps}$ factor) and `out_rms` keeps climbing — the accident scene of "forgot to enable the update / flipped the coefficient".

## Constraints

- CPU; float32 in, float32 out (accumulate internally in float64); $1\le T\le 64$, $1\le B\le 256$; `eps > 0`; `momentum ∈ [0,1]`.
- Variances are always population-style (divide by $n$, not $n-1$); std $=\sqrt{\text{var}+eps}$.
- Checking: `normalized`, `out_rms`, `running_mean`, `running_var` with `allclose(atol=1e-5, rtol=1e-5)`.
- Degenerate invariants: `whiten_std=0` ⇒ the output is exactly "mean removed" (with `stat_source=0` each batch has zero mean); equal rewards inside a group ⇒ that group's output is $0$ (zero numerator, `eps>0` prevents a division by zero); `momentum=1` ⇒ the ledger is exactly the previous batch's statistics; `momentum=0` ⇒ the ledger stays at its initial values.
- Division of labour: `to-grpo-advantage` derives the **formula** "group-standardize into an advantage" (one source, one batch); `to-gae` is the advantage recursion; `rl-trajectory-episode-assembly` assembles the reward tensors. This problem is only about **where the statistic comes from, when it is updated, and whether the update happens before or after normalization** — the pipeline bookkeeping itself.
