# notes (ag-memgpt-paging-ledger)

## Sourcing

- cpacker/MemGPT **301-redirects to letta-ai/letta** (the API returns 404/Moved without following). Paper-era code pinned to **tag 0.1.6** (oldest available tag, the pymemgpt era): `memgpt/agent.py` (AgentAsync.step / summarize_messages_inplace / trim_messages / prepend_to_messages), `memgpt/memory.py` (recursive summarize_messages), `memgpt/constants.py` (MESSAGE_SUMMARY_WARNING_TOKENS=**7000**), `memgpt/system.py` (package_summarize_message).
- curl against raw.githubusercontent kept getting SIGKILLed; WebFetch retrieved the text — but WebFetch output loses some indentation, so **line-by-line verification required semantic reconstruction** (the indent level of `cutoff = new_cutoff` is this problem's headline and cannot be judged by eye from a WebFetch dump; cross-check against Python semantics and multiple sources).

## Key official quirks (verified line by line)

1. **Role-shift bug (headline)**: `cutoff = new_cutoff` sits **outside** the inner `if messages[new_cutoff]["role"] == "user"` — the official log prints "Shifted cutoff ... is still a 'user', ignoring..." yet **the shift happens anyway**. Comment and behavior contradict; the eq-dead-inner-if mutation (deleting the dead inner if) SURVIVING is the proof. Only an out-of-range new_cutoff (IndexError) keeps the original cutoff.
2. **Smart cutoff runs down to -1**: break requires `>= 0.2*warning` and `cutoff -= 1` comes after — if never triggered, cutoff = len-1-len = **-1**; -1 participates in Python negative indexing: `messages[-1]` hits the last message (shifts to 0 if user), `messages[1:-1]` drops both ends, `trim_messages(-1)` = `[m0] + messages[-1:]`. For len<3 the cap (len-3) is negative; at len=1 `messages[-2]` raises IndexError inside the try → cutoff stays -2.
3. **System not exempt from accumulation**: the reversed loop accumulates tok(str(m)) over all messages (the whole dict repr); exemption only exists in the slice [1:cutoff] and trim's kept [0]. With cutoff=0, trim(0) keeps the full list plus prepend → **the system message appears twice in new_messages**.
4. **Recursive summary**: gate `tok(str(seq)) > warning_tokens` strictly greater (exact anchor 140==140 not triggering); ratio = (th/total)*0.8; cut2 = int(n*ratio) truncates toward zero (the round mutation caught at 2.38); the recursive **string** result mixes with the remaining dicts via `str([sub] + rest)` — repr lengths must be computed exactly (measured str(dict) base = 27 + len(role) + len(content); our first hand-derivation used 32+ and was corrected by printing — repr lengths must always be measured).
5. **Ledger**: hidden = all_time - remaining; remaining uses the sliced length (negative-index clamping: at len=1, messages[-2:] has 1 element); messages_total += len(added) (the summary counts as an addition).
6. **Packaging**: `"...\n" + f"The following is a summary of the previous {n} messages:\n {summary}"` — the space in `\n {summary}`; json.dumps of {"type": "system_alert", "message", "time"}.
7. The step-side memory-pressure alert (current_total_tokens > WARNING → agent_alerted_about_memory_pressure set, reset after summarize) is a boolean one-shot with thin signal — recorded here, not included.

## Engineering

- Inputs: tok / summarize (pure s->str) / time / warning_tokens all explicit; summarize's input is exactly the summary_input string fed to the official LLM (the ledger records every call input in order, recursion included — inner calls first, outer last).
- **All anchors computed from exact str(dict) reprs**: print len(str(m)) to pin the base (27+len(role)+len(content)) before writing tests; three anchors were wrong in the first draft purely on the base (35/39/43 vs 33/40/44), fixed by measurement — lesson: always measure repr lengths; formula derivation will be wrong once.
- Two exact anchors: accum_exact_break (last message's str exactly 39 = 0.2*195; `>=`→`>` distinguished via cutoff_pre_min because the post-min cutoff is masked by the keep-tail cap — you must assert pre_min) and recursion_exact_gate (140==140 gate not triggering plus a 139 case pinning cut2=2).
- Independent oracle: closure recursion + "".join segment assembly vs solution's f-strings; summarized indices rewritten as range expressions.
- Mutations: 25 substantive all CAUGHT + 3 equivalences recorded: eq-min-commute (min commutativity), eq-dead-inner-if (the official dead check), eq-accum-add-flip (addition commutativity).
- Red lines: openai calls/acreate, faiss archival, persistence_manager, the heartbeat/function_failed loop (free_step), persona/human char-limit edit gates, and CLI commands like /pop are not modeled.

## Follow-up candidates

- CoreMemory's persona/human edit gates (the 2000-char-limit error assembly + edit_replace exact matching) as an Easy sequel.
- The step loop's heartbeat/function_failed/token_warning three-way skip-input state machine for an agent-loop problem.
