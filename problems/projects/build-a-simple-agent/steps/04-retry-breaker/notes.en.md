# Step Notes

## Done Before

The ReAct loop yields this round's tool-call sequence, and the schema validator flags invalid calls.

## Introduced Here

Turn the failure path into deterministic data too: backoff is a set of integers inside a budget, the breaker is a windowed count, and an L2 assertion keeps the plan total within budget.

## What the Next Step Does With It

The final acceptance run records each turn's `stopped_by` as a breakpoint in the multi-turn report, asserted alongside compaction counts and tool-call sequences.
