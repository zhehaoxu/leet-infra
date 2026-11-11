# Step Notes

## Done Before

The schema validator is frozen as `scaffold.tool_schema` and can decide whether any single tool call is legal.

## Introduced Here

Turn a sequence of fixture replies into a trace with an explicit stop reason: the loop boundary, how invalid calls are recorded, and budget exhaustion are all pinned down.

## What the Next Step Does With It

The memory ledger turns this trace into replayable, compactable entries; retry/breaker reuses the same tool sequence. Both branches fork from here.
