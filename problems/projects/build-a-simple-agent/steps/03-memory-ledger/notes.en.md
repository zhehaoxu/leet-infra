# Step Notes

## Done Before

The ReAct loop produces a trace with an explicit stop reason, and tool-call legality is decided by the frozen step-01 module.

## Introduced Here

Introduce two ledger invariants: replaying the same fixture yields the same ledger, and the ledger stays bounded after compaction. The L2 checks target exactly these.

## What the Next Step Does With It

The final acceptance run writes each turn's trace into this ledger and combines it with step 04's retry breakpoints to form a multi-turn report.
