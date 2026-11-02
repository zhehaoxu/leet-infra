# notes — ag-memorybank-forget-ledger (order 722)

## Source of truth

- Official repo **zhongwanjun/MemoryBank-SiliconFriend** (`zhongwanjun/MemoryBank` is 404); the mechanism lives in `memory_bank/memory_retrieval/forget_memory.py`.
- **The precedence trap in `forgetting_curve(t, S) = math.exp(-t / 5*S)`**: Python evaluates it as exp(((-t)/5)*S) = **exp(−t·S/5)** — higher strength decays faster, exactly opposite to the paper/注释 intent exp(−t/(5S)). The official code wins; the divergence is recorded (and it is the strongest mutation anchor: the parenthesized "paper-fixed" version is caught by the curve anchor).
- Official stochastic forgetting `if random.random() > retention_probability: forget` — **strictly greater**, equality keeps; draws are explicit inputs aligned with the iteration order.

## Key ledger points

1. **Iteration order**: user insertion order → date order → dialog order; officially the loop iterates the **original** user_memory['history'] while pops hit the deepcopy — the equivalent implementation here snapshots `list(history.items())`; iterating the copy directly raises RuntimeError.
2. **Legacy migration writes back**: non-dict dialogs become `{query: d[0], response: d[1]}` in place.
3. **Default triple + `.update` write-back**: strength=1, last_recall_date=own date, memory_id=f"{user}_{date}_{i}" (the default includes the index; both memory-id-default mutations are caught by anchors).
4. **Reverse-order pop**: `forget_ids.sort(reverse=True)` before index pops (ascending pops shift — caught by the multi_forget anchor).
5. **Emptied dates sync-delete**: after `history.pop(date)` the summary pop is **unconditional** (official KeyError on missing key — ValueError here); the `if 'summary' in keys` check comes only afterwards for doc appending.
6. **Users without history dropped entirely**: the official `continue` fires **before** deepcopy — such users don't even keep their summary in the result bank.
7. **Summaries are never forgotten**: they consume no draw; non-dict values wrap into {'content': ...}; metadata written back (memory_id=f"{user}_{date}_summary"); summary docs appended for surviving dates.
8. **Reinforcement**: `id.split('_')[1]` locates the date (official direct indexing — missing date KeyError→ValueError); first matching id breaks; ids of emptied dates cannot hit (unless the date still holds other memories); `+= 1` and last_recall_date refresh.
9. **Doc strings**: cn `[|用户|]： {q.strip()}; [|AI伴侣|]： {r.strip()}` (**space after user_kw** — inside the f-string); en `Conversation content on {date}:` (**no space after the colon**, the space before q comes from f'{user_kw} '); summary cn `时间{date}的对话总结为：{content}`.

## Not modeled

- FAISS/embedding retrieval (recalled_ids explicit), LLM daily/global summarization, textsplitter chunking, file IO, and `search_memory`'s date grouping (whose source=memory_id never matches the page_content date prefix — the official replace is a no-op; contradictory upstream, excluded and recorded).

## Authoring lessons

- **The official code iterates the original dict while popping the copy** — the equivalent implementation must snapshot `list(history.items())`; iterating the copy directly raises "dictionary changed size during iteration" (caught by random on the first run).
- **forgetting_curve is not callable from tests** (the harness passes only solve) — the curve anchor asserts via the ledger's retention field instead.
- Summaries are popped together with emptied dates — the intuitive "only the summary doc remains" assertion was wrong; summary docs are appended only for **surviving** dates.
- All 17 substantive mutations CAUGHT (update-no-break first showed PATTERN NOT FOUND due to indentation, re-verified after fix) plus 1 equivalence (the redundant list() snapshot over summary keys — the loop only rebinds values, SURVIVED confirmed).
