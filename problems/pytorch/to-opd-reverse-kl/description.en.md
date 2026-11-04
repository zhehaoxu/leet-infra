# OPD: Reverse-KL On-Policy Distillation Accounting

## Problem

Conventional distillation fits the student to the teacher's forward
distribution (forward KL / JSD); OPD (On-Policy Distillation) flips it — the
**student samples on-policy first**, the teacher scores the student's actual
rollouts token by token, `reverse KL = student_logprob − teacher_logprob`:
tokens the student prefers more than the teacher get penalized, less-preferred
ones get rewarded, and `−reverse_kl` serves directly as the advantage for an
importance_sampling update. No value net, no reward model, no group baseline.
This problem replicates the deterministic accounting of agentic-rl-lab
(PyTRIO/datasets/swanlab stripped, rollouts and teacher scoring fully
parameterized) via four `solve(mode=...)` entries:

1. **teacher_logprobs**: compute_logprobs returns per-token logprobs for the
   whole all_ids sequence; the completion region is sliced from
   `len(prompt_ids)` to the end; the length must equal
   `len(completion_ids)` exactly and contain no None → else ValueError.
2. **datum**: right-shift four fields — `prompt_loss_len =
   len(prompt_ids)−1`; input = prompt + completion[:-1]; target =
   [0]×prompt_loss_len + completion; logprobs/advantages zero-padded up
   front; equal lengths or ValueError; dtypes int64 / float32.
3. **step**: batch ledger — empty sequences skipped; `reverse_kl =
   student_lps − teacher_lps` per-token vector difference; `advantages =
   −kl_penalty_coef × reverse_kl`; reverse_kls extended per token; empty
   datums → RuntimeError; metrics = mean/std (ddof=0).
4. **schedule** (IDT interleaved distillation): build the cycle tuple;
   resolve with steps=0 uses each task's ceil(size/batch) steps, takes the
   max cycle count, and multiplies back; count via `schedule[step % len]`;
   batch wraps with each task's own step index.

## I/O

- `teacher_logprobs`: all_logprobs + prompt_ids + completion_ids → slice
- `datum`: four params → `{input_ids, target_tokens, logprobs, advantages}`
- `step`: batches + teacher_fn + kl_penalty_coef → 4-tuple
- `schedule`: variant ∈ build/resolve/count/batch

## Example

A 5-token prompt + 4-token completion:

- teacher slice: all_lps[5:] = the completion region
- datum: input length 8 (5+3), target = [0,0,0,0,201,202,203,204],
  logprobs/advantages zero for the first 4 positions
- IDT 2:1 cycle over 7 steps → medical 5 / ceval 2

## Edge cases and pitfalls

- Teacher length mismatch falls into the post-slice length check →
  ValueError, never IndexError.
- The float32 contract carries noise (−0.6000000238…); tests align via
  round(·,6).
- Statistics are token-granular: skipped empty sequences produce neither
  datums nor statistics.
- The IDT batch uses each task's own step index — the two datasets never
  skip each other's steps.
