# MemGPT Main-Context Paging Ledger (ag-memgpt-paging-ledger)

**Difficulty**: Hard ｜ **Collection**: agentic-systems ｜ **Order**: 726

A stateless rebuild of MemGPT's (arXiv **2310.08560**, repo now [letta-ai/letta](https://github.com/letta-ai/letta); the paper-era **tag 0.1.6** is authoritative) main-context paging core: `memgpt/agent.py`'s `summarize_messages_inplace` (smart cutoff → keep-tail min → role shift → slice → recursive summary → ledger/packaging → trim/prepend) and `memgpt/memory.py`'s recursive `summarize_messages`, line by line. The summary LLM, token counter, and timestamp are explicit inputs.

## What you implement

`solve(messages, all_time_message_count, warning_tokens, tok, summarize, time)`: perform one paging pass over the main context, returning the three cutoff stages, the recursive call-input order, the packaged summary message, and the trim/prepend end state.

## Ledger points (verified line by line against tag 0.1.6)

1. **Smart cutoff**: `tokens_so_far=0`, `cutoff=len(messages)-1`; over `reversed(messages)` accumulate `tokens_so_far += tok(str(m))` (**the whole dict repr**, not the content — the system message is NOT exempt), break only **after** accumulating when `>= warning_tokens * 0.2`, else `cutoff -= 1` — if never triggered, the loop runs out and cutoff reaches **-1**.
2. **Keep-tail**: `cutoff = min(len(messages) - 3, cutoff)` (official comment "Always keep the last two messages too"); for len<3 the cap is negative and interacts with -1 to produce degenerate quirks.
3. **Role shift (official bug, the headline)**: `if messages[cutoff]["role"] == "user"` then `new_cutoff = cutoff + 1` — the inner `if messages[new_cutoff]["role"] == "user"` **only prints "ignoring"**, and `cutoff = new_cutoff` sits **outside** the inner if, so the shift happens **unconditionally**; only an out-of-range `new_cutoff` (IndexError) keeps the original. With `cutoff=-1`, `messages[-1]` hits the last message (negative indexing).
4. **Slice**: `seq = messages[1:cutoff]` — the system message (index 0) is exempt **by position** (no role check); cutoff=0 → empty sequence (`str([])='[]'`); cutoff=-1 → `[1:-1]` drops both ends.
5. **Recursive summary**: when `tok(str(seq)) > warning_tokens`: `trunc_ratio = (warning_tokens / total) * 0.8` ("For good measure"), `cut2 = int(len(seq) * ratio)` (truncate toward zero), `summary_input = str([recursive(first cut2)] + rest)` — the recursive **string** result is mixed with the remaining dicts in one list then `str()`-ed; the gate is **strictly greater** (equality does not trigger).
6. **Ledger & packaging**: `remaining = len(messages[cutoff:])` (negative-index Python semantics); `hidden = all_time - remaining`; the summary message is `json.dumps({"type": "system_alert", "message": "Note: prior messages ({hidden} of {total} total messages) have been hidden from view due to conversation memory constraints.\nThe following is a summary of the previous {n} messages:\n {summary}", "time": time})` — format copied verbatim (including the space in `\n {summary}`).
7. **Trim + prepend**: `trim_messages(cutoff)` (= `[messages[0]] + messages[cutoff:]`, keeps the system message — with cutoff=0 the whole list is kept so the system message appears twice), then prepend `{"role": "user", ...}`; `messages_total_new = all_time + 1` (the summary counts as an addition).

## Example

```python
msgs = [{"role": "system", "content": "S0"},
        {"role": "user", "content": "x" * 8},
        {"role": "user", "content": "y" * 8},
        {"role": "assistant", "content": "z" * 3}]
led = solve(msgs, 6, 200, len, lambda s: "S<%d>" % len(s), "T")
# threshold 0.2*200=40: last 39<40, next 78>=40 → break, pre_min=2
# min(1,2)=1; messages[1] is user → shift to 2 (even though messages[2]
# is also user — "ignoring" has no effect) → cutoff=2, seq=[msgs[1]]
```

## Scoring

10 checkpoints: negative cutoff and negative-index shifting, the "ignoring" dead-check bug, len<3 degeneracies (-1/-2/IndexError), the exact recursion gate (140==140 not triggering), cut2 truncation and the mixed-list string, the exact accumulation break point, hidden/remaining bookkeeping, trim/prepend end state (duplicated system), packaging format, and 150 randomized main-context cross-checks.
