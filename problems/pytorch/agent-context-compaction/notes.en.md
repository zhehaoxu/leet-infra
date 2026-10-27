# Solution Notes · Context Budget Compaction

## Compaction is a knapsack with pinned items

Every segment has a weight (tokens) and a value (priority); the window is the capacity. 0/1 knapsack is NP-hard, but production systems never solve it exactly: priority is a human-supplied value order, and greedy is good enough and explainable. What must really hold is **determinism** — the same input has to produce the same keep-set, otherwise reproducing an incident is hopeless.

Pinned segments are a "must be packed" constraint: subtract their weight first, then run ordinary greedy on the rest. If even they do not fit, the configuration itself is wrong (window too small, system prompt too long) and that must be an error, not a silent degradation.

## Why skipping is not terminating

Two classic mistakes:

- `break`: assuming that whatever comes after the first non-fitting item also will not fit — but the order is by priority, not by size, so a later small segment may well fit;
- an infinite loop: skipping without advancing the cursor.

The correct shape is a single pass in priority order, packing what fits and passing over what does not, with `used` monotonically non-decreasing — O(n log n), dominated by the sort.

## Why the output is chronological

Prompt assembly must follow conversational time: system first, history in the middle, scratchpad last. Assembling in priority order hands the model a shuffled context where conclusions precede evidence, which is worse than dropping segments. So selection uses priority and the output uses original indices.

## More dials in real systems

- **Recency bonus**: nearer segments are likelier to survive, usually as `priority + k / age`;
- **Summary slot**: a dropped middle section becomes a "summarised" placeholder that itself costs a fixed number of tokens;
- **Pair constraints**: `tool_call` and `tool_result` must travel together or the protocol is invalid;
- **Token estimation error**: offline counting uses the tokenizer, online code often uses heuristics, so a safety margin is needed.

This problem keeps only the "pinned + priority + budget" spine because it is the most judgeable — and where incidents actually happen.

## Related problems

- `agent-react-loop`: the longer the loop, the more segments need compacting.
- `agent-tool-call-schema`: tool schemas are themselves the biggest thing being compacted.
- `agent-task-dag-schedule`: the scheduling counterpart for parallel sub-tasks.
