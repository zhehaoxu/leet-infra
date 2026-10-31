# notes — rs-budget-forcing (order 662)

## Source

- Origin: the budget-forcing technique from s1: Simple test-time scaling (2501.19393): when the `<think>` segment hits the token budget, inject `</think>Final Answer:` (length forcing); to extend thinking before the budget is spent, inject `Wait` (wait forcing, multi-round resumption). The official implementation is tied to a decoding stream; this problem abstracts the observable part into a **token-accounting state machine**: step sequences, budget, wait cost, answer truncation.
- Three error-prone points pinned by the spec:
  - a truncated thinking step records its **prefix length** in `step_emitted` (not 0, not the full step);
  - a Wait step pays `wait_cost` first and then emits; Wait tokens count toward the total but **never into `think_emitted`**;
  - failing to afford `wait_cost` stops the attempts (it does not skip and try later Waits).

## Judging

- Oracle: `test_cases._py` is an independent rewrite (a while loop deducting token by token), 200 random cross-checks (random budget/step/wait combinations).
- Anchors: exact-boundary step `emitted + c == budget` is emitted whole; truncated step records the prefix (40, not 55: wait tokens go into `wait_emitted`, not `think_emitted`); unaffordable wait_cost stops with `wait_log=False`; truncation after a Wait stops later Waits; the answer truncates to remaining + `answer_truncated`; with no Waits it degenerates to plain length forcing.
- Invalid inputs: non-list sequences, bool/negative/non-int entries, non-positive-int budget, bool/negative wait_cost → ValueError.

## Mutation log (9/9 caught)

budget-off-by-one (`<=` → `<`), forced-not-set, wait-cost-free (wait_cost never deducted), wait-after-forced (Waits continue after truncation), final-no-truncate, wait-not-counted (wait deduction lost), step-prefix-dropped (truncated step records 0), think-total-wait (wait mixed into think_emitted), first-wait-free (first Wait free).

## Red lines

No model calls or real decoding streams; boundary with `rs-overthinking-metrics` (efficiency statistics): this is a **budget-control state machine**, not cross-sample statistics.
