# PEFT: GenKnowSub General-Knowledge Subtraction

## Problem Description

**GenKnowSub** (arXiv:2505.10939, ACL 2025, official `saharsamr/Modular-LLM`; on the PEFT side it shares arrow.py via `use_gks`/`gks_` prefixes) hypothesizes that task LoRAs are **entangled with general knowledge**, harming zero-shot generalization: build one (or more) **general-domain LoRAs**, **average their A/B factors**, and **subtract the averages from every task adapter** ("forgetting-via-negation" task arithmetic, arXiv:2212.04089) to obtain purified residual task modules, then route them with Arrow. The subtraction acts on the **factors**, so the post-subtraction task delta \((B-\bar B)(A-\bar A)\) carries cross terms — it is NOT \(B@A-\bar B@\bar A\).

This problem replicates the deterministic account of the official PEFT `gen_know_sub` + `create_arrow_model` gate:

- **General average**: `avg_A = stack(A_gks).mean(0)`, `avg_B = stack(B_gks).mean(0)` (element-wise mean; multiple language GK LoRAs participate).
- **Factor subtraction**: the official in-place `lora_A[name].weight.data.sub_(avg_A)` (same for B); in the judging domain a pure function — clone then subtract. The first run (`gks_done=False`) purifies **all** task adapters; incremental runs only the newly added ones (`gks_added_adapter_names`).
- **Gate state machine**: `use_gks=False` -> skip; `use_gks ∧ gks_done ∧ nothing added` -> skip; `use_gks ∧ ¬gks_done` -> all; `use_gks ∧ gks_done ∧ added` -> added.
- **Path validation**: `use_gks=True` with no general LoRA paths -> ValueError ("You should provide general LoRA paths if you want to use GenKnowSub.").

## Task

- `mode="gks_average"`: inputs g_a, g_b (GK factor lists); return (avg_A (r,in), avg_B (out,r)).
- `mode="gks_subtract"`: inputs a_stack, b_stack, avg_a, avg_b, only (optional index subset); return the purified (a_new, b_new); the input stacks stay untouched.
- `mode="gks_gate"`: inputs use_gks, gks_done, n_added; return "skip"/"all"/"added".
- `mode="require_gks_paths"`: inputs use_gks, n_paths; return n_paths; invalid combinations raise ValueError.
- `mode="task_delta"`: inputs a_sub, b_sub; return the per-expert \((out,in)\) delta stack.

## Example

seeds 0/1/2, float64:

```
gks_average(g_a [3,2,8]×3, g_b [3,6,2]×3) -> avg_a (2,8), avg_a[0,0] = -0.7495449019423001
gks_subtract(a (4,2,8), b, avg_a, avg_b) -> a - avg_a per expert; a_new[0,0,0] = 0.438255347752268
gks_subtract(..., only=[1,3]) -> only experts 1/3 purified, others bit-identical
gks_gate(True, False, 0) == "all"; gks_gate(True, True, 0) == "skip"; gks_gate(True, True, 2) == "added"
require_gks_paths(True, 0) -> ValueError
task_delta(a_new, b_new)[0,0,0] = 2.1708577088384393
  and d == B@A - B@avgA - avgB@A + avgB@avgA (cross terms), ≠ B@A - avgB@avgA
```

## Constraints

- Averaging and subtraction both act on the **factors** (A/B), not on dW — cross terms cannot be dropped.
- `gks_subtract` is a pure function: the input stacks must stay bit-identical (the judging-domain equivalent of the official in-place `data.sub_`).
- Gate priority: `use_gks=False` always skips (overriding gks_done/added).
- With an `only` subset the non-selected experts are **bit-identical** (`torch.equal`), not approximately equal.
- The cross-term expansion of the task delta: all three terms \(B\bar A + \bar B A - \bar B\bar A\) are nonzero.
