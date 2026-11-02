# ReWOO Placeholder Plan Ledger

## Problem

ReWOO (Xu et al., 2305.18323) removes the redundancy of ReAct-style interleaving: the Planner emits the whole plan at once, Workers execute the tool calls, and the Solver sees all evidence only at the end. The official implementation (billxbf/ReWOO) keeps this in `algos/PWS.py`.

The planner text, each tool's real return, and the solver's final answer are non-deterministic calls, so this problem takes them **entirely as explicit inputs** and tests three deterministic sections:

- **Plan parsing**: lines starting with `Plan:` enter plans; `#E<d> = ...` lines split on the first `=` (both sides stripped), and the evidence id must be exactly `#Ed` (single-digit, len==3) — `#E12 = Google[q]` is recorded as `No evidence found`, the tool call discarded.
- **Worker substitution** (in evidence insertion order): a declaration without `[` passes its raw text through as evidence; with `[`, split on the first bracket for the tool name, **blindly drop the last character** of the rest (conventionally `]`), then replace every `#E\d+` placeholder with `[evidence]` in original-occurrence order — **placeholders appearing inside already-inserted evidence are hit by later replacements** (official replace-all quirk); forward references stay literal. Tools outside `tools` yield evidence `No evidence found`; Google counts calls, LLM counts `len(call+evidence)//4` tokens.
- **Assembly and cost**: plan i pairs with evidence `#E{i+1}` **by index** to build `worker_log` (surplus evidence excluded); `token_cost = planner_price*planner_tokens + solver_price*solver_tokens + fixed davinci-003 price*LLM tokens`; `tool_cost = Google_count*0.01`; `steps = len(plans)+1`.

Price table (official `get_token_unit_price`): text-davinci-003=0.00002, gpt-3.5-turbo=0.000002, gpt-4=0.00003, the two alpaca-lora names=0.0, unknown names raise `ValueError`.

## Solution

Implement `solve(planner_response, worker_outputs, solver_output, planner_tokens=0, solver_tokens=0, tools=("Google","LLM"), planner_model=..., solver_model=...)` returning the ledger dict (plans/planner_evidences/worker_calls/worker_evidences/worker_log/output/tool_usage/steps/total_tokens/token_cost/tool_cost/total_cost). Domain constraints: an evidence line without `=`, a valid tool call without a worker output, or a plan line without its paired evidence → `ValueError`.

## Example

```python
resp = "Plan: a\n#E1 = Quote #E2 line\n#E2 = Google[q]\n#E3 = LLM[#E1 and #E2]\n"
solve(resp, {"#E2": "cats", "#E3": "ans"}, "ans")["worker_calls"]["#E3"]
# == "[Quote [cats] line] and [cats]"
# #E1 expands to "[Quote #E2 line]"; the later #E2 replace-all hits that inserted #E2 too
```

## Constraints

- At most 6 steps, evidence ids 1-12 (multi-digit quirk included), short strings; 200 seeded random cross-checks plus exact anchored assertions.
