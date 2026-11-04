# OPD Solution Notes

## Key official details (pinned by sandbox runs)

1. **Dual nature of the teacher slice check**: the official code slices
   `completion_logprobs = all_logprobs[len(prompt_ids):]` then checks
   `len != len(completion_ids) or any None` — a short all_logprobs yields a
   short slice falling into the same ValueError, never IndexError.
2. **Asymmetry of the right-shift ledger**: input takes completion[:-1],
   target takes the full completion — both of contract length
   len(prompt)−1+len(completion); off-by-one shift mutants are pinned by
   the exact [0]×4+completion target.
3. **float32 is the PyTRIO schema contract** (logprobs/advantages float32,
   targets int64) — expectations carry float32 noise (−0.60000002384…);
   tests compare via round(·,6).
4. **Statistics granularity**: reverse_kls extends per token — mean/std
   cover the whole batch at token granularity (6 tokens in the sample:
   mean −1/6, std of that order); skipped empty sequences produce neither
   datums nor statistics.
5. **Empty datums raise RuntimeError**: the official code separates "no
   trainable data" (RuntimeError) from "malformed data" (ValueError).
6. **Three wrap-arounds in the IDT schedule**: schedule[step % len] picks
   the task, rows[(start+offset) % len(rows)] picks the batch, and resolve
   multiplies the max cycle count back. 7 steps of a 2:1 cycle → m5/c2
   (step 7 wraps to medical).
7. **batch_for_task_step uses each task's own step index**: medical step 0
   and ceval step 0 both start at their own rows[0] — distinguishable from
   a global-step-index mutant.
8. **resolve's steps=0 semantics**: guarantee both datasets are traversed
   at least once — ceil(size/batch) steps each, max over the cycle shares,
   multiplied back by the cycle length.

## Boundary with neighboring problems

- `to-gkd-jsd-loss`: GKD fits the teacher's forward distribution via JSD
  (needs teacher forward logits/distributions); OPD only needs the
  teacher's per-token logprobs of concrete sequences, the KL direction is
  student−teacher, and it is used directly as the advantage.
- `to-kl-controller`: RLHF's KL penalty is a per-token penalty against a
  reference policy inside the pg loss; OPD's "KL" is the distillation
  signal itself — no reference policy, no PPO clip.

## Sandbox lessons

- When extracting the step kernel, **count the nesting depth**: the
  re-indented kernel got only 4 spaces of prefix, landing `for seq` at the
  same level as the batch loop after it → empty for-body IndentationError —
  re-indent kernels to their target depth (8 spaces here).
- **Type annotations are dependencies too**: `args: argparse.Namespace` is
  evaluated at exec time (stub argparse in the env); `trio.ModelInput.
  from_ints` inside a function body needs a trio stub.
- The first solution draft used dict access (seq["tokens"]) while the
  fixture exposes objects (seq.tokens) — the official surface is attribute
  access; follow the official code.
- Round(·,6) the float32 noise before writing EXPECTED, and round on the
  test side as well.
