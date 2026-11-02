# MemoryBank Forgetting-Curve Ledger

## Problem

MemoryBank (Zhong et al., 2305.10250) equips an LLM companion with long-term memory driven by the Ebbinghaus forgetting curve: memories decay over time and are reinforced when recalled. The official implementation (zhongwanjun/MemoryBank-SiliconFriend) keeps the mechanism in `memory_bank/memory_retrieval/forget_memory.py`.

The memory bank, current date, and retrieval hits are external state, so this problem takes them **entirely as explicit inputs** (including the random draw per memory), testing two sections:

- **Forgetting** (`initial_load_forget_and_save`): iterate user insertion order → date insertion order → dialog order; legacy `[query, response]` dialogs migrate to dicts; defaults `memory_strength=1`, `last_recall_date=own date`, `memory_id=f"{user}_{date}_{i}"` written back via `.update`; retention `= math.exp(-t / 5*S)` — **the official precedence trap: exp(−t·S/5), so higher strength decays faster** (opposite of the paper's intent exp(−t/(5S)); official code wins); a memory is forgotten iff `draw > retention` (strict, equality keeps); forgotten indices pop in **reverse order**; an emptied date is removed from history and summary **together**; a user without a `history` key is **dropped from the result bank entirely** (official continue placement).
- **Reinforcement** (`update_memory_when_searched`): locate the date via `memory_id.split('_')[1]` (missing → official KeyError, here ValueError), first matching `memory_id` gets `strength+1` and `last_recall_date=cur_recall_date`, then break (duplicates update only the first; memories in emptied dates cannot be hit).
- **Document ledger**: kept dialogs yield `时间{date}的对话内容：[|用户|]： {query.strip()}; [|AI伴侣|]： {response.strip()}` (en: `Conversation content on {date}:[|User|]: ...`); summaries are never forgotten — non-dict values wrap into `{"content": ...}` with metadata written back (`memory_id=f"{user}_{date}_summary"`) — and yield summary documents.

## Solution

Implement `solve(bank, now_date, draws, recalled_ids=(), user=None, cur_recall_date=None, language="cn")` returning `{"bank", "dialogs", "docs", "updates"}`. Domain constraints: dates are `YYYY-MM-DD`, draws are numbers in [0,1], `language ∈ {cn, en}`; insufficient draws, missing summary key, unknown recall date → `ValueError`.

## Example

```python
bank = {"u": {"history": {"2026-01-01": [
    {"query": "q", "response": "r", "memory_strength": 7,
     "last_recall_date": "2025-12-31"}]},
    "summary": {"2026-01-01": "s"}}}
led = solve(bank, "2026-01-01", [0.5])
led["dialogs"][0]["retention"]  # ~ exp(-1.4) ~ 0.2466 (exp(-t*S/5))
led["dialogs"][0]["forgotten"]  # True: 0.5 > 0.2466
led["docs"]                     # []: emptied date pops the summary too
```

## Constraints

- At most 3 users, dates within a 9-day window, at most 3 dialogs per user; 200 seeded random cross-checks plus exact anchored assertions.
