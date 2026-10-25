# Deep dive: The Reward-Normalization Batch-Statistics Pipeline

## Origin and background
- OpenRLHF calls `reward_norm(...)` right after rollout. Its `group_norm` switch has five settings —
  `prompt / group / running_state / batch / none` — i.e. **"which statistic do I divide by" is a scheduling
  option, not a mathematical one**; a `RunningMoments` object feeds the `running_state` setting. verl's
  side is the same mechanism under advantage/reward normalization switches (GRPO takes the group source;
  switches decide whether the std is divided out at all).
- TRL's PPO trainer likewise uses `RunningMoments` to normalize both the KL term and the rewards, with a
  default discount `gamma=0.99`.
- What this problem extracts is the **data flow**: $T$ batches in arrival order, each one doing
  "pick a statistic source → normalize → write the ledger", plus an `out_rms` probe that turns a silent
  scale-mismatch incident into a visible number. Everything is add/multiply/divide over given tensors;
  no advantage or loss derivation appears anywhere.

## Four conventions to pin down
1. **Update order: normalize first, then write the ledger.** With `stat_source=1` batch $t$ reads
   `running_mean[t] / running_var[t]`, i.e. the state *before* batch $t$ is written; the ledger is only
   updated in step 4. This matches BatchNorm train/eval semantics and explains why "the very first batch is
   effectively un-normalized" (`normalized[0]` is just $x/\sqrt{1+eps}$ in the example). Updating before
   reading shifts the whole timeline by one and fails the cross-check immediately.
2. **The ledger is always driven by whole-batch statistics**, independent of `stat_source`.
   `stat_source` answers "what do I divide by", not "what do I write down". All three sources in the
   example produce identical `running_mean/running_var`; the test `scale_mismatch_probe_fires` asserts
   exactly this.
3. **The momentum convention is PyTorch BatchNorm's $m$, not TRL's $\gamma$**: here $r\leftarrow(1-m)r+m\,x$,
   while TRL/OpenRLHF's `RunningMoments` uses $r\leftarrow\gamma r+(1-\gamma)x$, so $\gamma=0.99$ equals
   $m=0.01$ here. Copying someone else's $\gamma$ straight into $m$ makes the ledger converge 100x faster
   and lets the earliest batches pollute the scale — the classic incident in this family. TRL's variance
   update also adds a mean-shift correction term (of the form
   $(1-\gamma)\big(v_{new}+\gamma(\bar x_{old}-\bar x_{new})^2\big)$) and uses `var(unbiased=True)`;
   this problem **deliberately omits both and standardizes on the population variance (ddof=0)** so that
   "divide by $n$ or by $n-1$" is stated on the problem face instead of hidden inside a library call.
4. **Group statistics are within-batch**: `group_ids` has length $B$ and every batch reuses the same group
   numbering (a group is not a global sample id). A group of size 1, or a group whose rewards are all equal,
   makes numerator and variance zero at once, so that entry is **exactly 0** (not $\pm1$ — you are dividing
   $0$ by $\sqrt{eps}$). Sparse group ids such as `[3,1,3,1]` are harmless: unused ids are never indexed.

## Why `out_rms` belongs in the output
$\sqrt{\operatorname{mean}(z^2)}$ sits at $\approx1$ whenever whitening really happened, and falls back to the
raw magnitude when `whiten_std=0`. It exposes three failure modes earlier than the loss curve does: wrong
source (`stat_source=1` meeting a reward-scale jump — 23x already in the statement's example), reversed
momentum (`momentum=0` freezes the ledger and `out_rms` climbs), and a badly chosen eps magnitude. In a real
trainer this is the normalized-reward std ratio you put on a dashboard.

## Common pitfalls
- Reading the ledger after updating it (convention 1).
- Using the sample variance: at $B=4$ the scale differs by $\sqrt{3/4}$, turning $\pm1.341635$ into
  $\pm1.161895$ in the example — an instant mismatch.
- Accumulating group statistics globally across batches: groups here are per-batch and recomputed each time.
- Dividing by std when `whiten_std=0`, or adding `eps` outside the square root.
- Forgetting that `running_*` has length $T+1$ (index 0 is the initial $0.0/1.0$; entries $1..T$ are the
  post-update states).
- Special-casing zero variance into something other than exact $0/\sqrt{eps}=0$.

## Division of labour with other families
- `to-grpo-advantage` (order 213) derives the **formula** "standardize inside the group into an advantage":
  one batch, one source, $\frac{r-\mu_g}{\sigma_g}$. This problem never defines an advantage; it only asks
  where the statistic comes from, when it is updated, and whether the update precedes normalization — with
  three interchangeable sources (batch / ledger / group).
- `to-gae` (order 211) is the temporal recursion; `rl-trajectory-episode-assembly` builds the reward tensor
  and applies the bootstrap rule. This problem starts where assembly ends.
- `to-kl-controller` and `to-reward-model-bradley-terry` are also formula-side (KL coefficient adaptation
  and the reward-modeling objective).
- `batchnorm-forward` (order 16), `groupnorm` (order 19) and `layernorm-forward` normalize along a **spatial**
  axis (features or channel groups) and compute batch statistics on the spot in training mode. Here the axis
  is **arrival time**: the crux is the EMA ledger and "normalize new data with old statistics", which is
  precisely what separates an RLHF reward pipeline from inference-time BatchNorm.
- `rl-per-priority-replay` uses reward-derived priorities for sampling probabilities and IS weights; this
  problem stops at "get rewards onto a unit scale".

## Complexity
$O(T\cdot B)$: one pass for the mean, one for the variance, one to write the output per batch; group
statistics use `index_add_` in $O(B)$ then broadcast by group id. Accumulate in float64, emit float32.
The oracle is pure-Python double loops with `math.fsum` and a dict-of-groups, a structurally different code
path so the two must agree by derivation rather than by copy.
