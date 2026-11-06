# Project Overview: Six Blocks, One Complete Data Flow

> This page unlocks after step 05 (the final acceptance run). It is the cognitive asset you take away: each problem only showed a local piece, here the whole flow is assembled.

## The Complete Data-Flow Puzzle

```text
[ user turn ]
      |
      v
[ fixture LLM replies ]  -- explicit args, no model in sandbox
      |
      v
[ 01 validate(spec, args) ]  -> ok / missing / unexpected / type_errors / normalized
      |
      v
[ 02 run_loop(script, tools, max_steps) ] -> trace / steps_used / terminated_by / final
      |
      +---------------------------+---------------------------+
      v                                                       v
[ 03 append / compact ]                         [ 04 plan(outcomes, policy) ]
   ledger, compactions                            delays / breaker / stopped_by
      |                                                       |
      +---------------------------+---------------------------+
                                  v
[ 05 run_dialog(turns, tools, policy) ] -> per-turn calls, breakpoint, final_answer
                                  v
[ 06 audit_stage_order(order) ] -> accepted / reason
```

## The Six Blocks You Actually Wrote

| Step | Your block | Earlier official scaffold | What it feeds |
| --- | --- | --- | --- |
| 01 | Schema validation | — | An assertable boundary for the loop |
| 02 | Loop state machine | tool_schema | Turn fixture replies into a trace |
| 03 | Memory ledger | tool_schema, react_loop | Turn traces into a replayable, compactable ledger |
| 04 | Retry and breaker | tool_schema, react_loop | Turn failure paths into a deterministic plan |
| 05 | Final assembly | all four earlier blocks | Assert multi-turn call sequences and answers |
| 06 | Design review | all five earlier blocks | Judge whether the decision order is sound |

## Three Takeaways

- **State travels through the scaffold chain, not through the judge**: once the judge replays a learner's earlier code, the state space explodes and anti-cheating collapses. Freezing earlier reference implementations keeps every step a single execution with a single input.
- **What is gradable in an agent project is the pure logic**: model output is not reproducible, but schema validation, loop termination, ledger compaction, backoff budgets, breaker windows and call sequences are. Externalise all generation into fixtures and the problem becomes stable.
- **Training steps must not judge "convergence"**: in a "train a tiny LM" project, L3 fixes the seed and batch, runs N steps, and only checks a loss band plus a parameter snapshot tolerance.

## What Next

Return to the project page to redo any step, or compare the official modules under `scaffold/` with your implementation. Project progress is kept separately in `data/project_progress.json` and never touches the normal problem bank's `progress.json`.
