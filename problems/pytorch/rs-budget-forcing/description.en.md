# Budget Forcing Reasoning Ledger: Truncation / Wait Resumption / Answer Truncation

## Problem

s1's budget forcing is an inference-time control: give the chain of thought a token budget, and when it is exhausted hard-switch the model to the final answer (`</think>` + forced final). This problem transcribes the ledger of that state machine: given a sequence of thinking steps `think_steps` (tokens per step), a `budget`, a sequence of wait steps `wait_steps` (tokens of the resumption prompt) with `wait_cost`, and the answer tokens `final_tokens`, emit the accounting records of the three-stage state machine.

## Solution

The three-stage state machine is pinned by the spec:

1. **Thinking stage**: emit `think_steps` in order. If `emitted + c <= budget` the whole step is emitted; otherwise emit `budget - emitted` tokens, set `forced_final=True`, and stop thinking (length forcing: the cut step contributes only its prefix, recorded in `step_emitted`).
2. **Wait stage** (only when not truncated): try `wait_steps` in order. Each step first requires `remaining >= wait_cost` (afford the resumption prompt); if affordable, deduct `wait_cost` and emit that Wait step per rule 1 (it may truncate again; after truncation no further Waits are attempted); if not affordable, stop trying (that step logs `wait_log=False`, emitting 0). If already truncated, all Waits are skipped.
3. **Answer stage**: the total of `final_tokens` is emitted truncated to `remaining`; if cut, `answer_truncated=True`.

Returns `{"step_emitted", "think_emitted", "forced_final", "waits_used", "wait_log", "wait_emitted", "final_emitted", "answer_truncated", "total"}`; `think_emitted` counts only the thinking stage (Wait tokens go into `wait_emitted` and the total, never into `think_emitted`).

## Constraints

- 1 ≤ budget ≤ 10^6, sequence length ≤ 10^4, token counts are non-negative ints.
- Non-list think_steps/final_tokens/wait_steps, bool/negative/non-int entries, non-positive-int budget, bool/negative wait_cost → `ValueError`.
