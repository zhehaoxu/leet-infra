# MemGPT 主上下文换页账（ag-memgpt-paging-ledger）

**难度**：Hard ｜ **合集**：agentic-systems ｜ **order**：726

把 MemGPT（arXiv **2310.08560**，仓今为 [letta-ai/letta](https://github.com/letta-ai/letta)，以论文同期 **tag 0.1.6** 代码为准）的主上下文换页核无状态化：`memgpt/agent.py` 的 `summarize_messages_inplace`（smart cutoff → 保尾 min → 角色移位 → 待摘切片 → 递归摘要 → 记账打包 → trim/prepend）与 `memgpt/memory.py` 的递归 `summarize_messages` 逐行记账。摘要 LLM、token 计数、时间戳全部显式入参。

## 你要实现的

`solve(messages, all_time_message_count, warning_tokens, tok, summarize, time)`：按官方口径对主上下文做一次换页，返回 cutoff 三态、递归调用输入序、摘要包与 trim/prepend 终态。

## 账点（全部来自官方 0.1.6 源码逐行核对）

1. **Smart cutoff**：`tokens_so_far=0`、`cutoff=len(messages)-1`；`reversed(messages)` 逐条 `tokens_so_far += tok(str(m))`（**整个 dict 的 repr**，非 content——system 不豁免），**累加后**判 `>= warning_tokens * 0.2` 才 break，否则 `cutoff -= 1`——从未触发时跑满减到 **-1**。
2. **保尾**：`cutoff = min(len(messages) - 3, cutoff)`（官方注释 "Always keep the last two messages too"）；len<3 时 cap 为负，与 -1 相交产生退化 quirk。
3. **角色移位（官方 bug，头号考点）**：`if messages[cutoff]["role"] == "user"` 则 `new_cutoff = cutoff + 1`——内层 `if messages[new_cutoff]["role"] == "user"` **只打印 "ignoring"**，`cutoff = new_cutoff` 在内层 if **之外**，移位**无条件发生**；仅 `new_cutoff` 越界（IndexError）才保持。`cutoff=-1` 时 `messages[-1]` 命中末条（负索引）。
4. **待摘序列**：`seq = messages[1:cutoff]`——system（index 0）**按位置**豁免（官方不查 role）；cutoff=0 → 空序列（`str([])='[]'`）；cutoff=-1 → `[1:-1]` 丢首尾。
5. **递归摘要**：`tok(str(seq)) > warning_tokens` 时 `trunc_ratio = (warning_tokens / 总量) * 0.8`（"For good measure"）、`cut2 = int(len(seq) * ratio)`（向零截断）、`summary_input = str([递归(前 cut2 条)] + 后段)`——递归返回**字符串**与剩余 dict 混装进一个列表再 `str()`；门是**严格大于**（恰等不触发）。
6. **记账与打包**：`remaining = len(messages[cutoff:])`（负索引同 Python 语义）；`hidden = all_time - remaining`；summary 包 `json.dumps({"type": "system_alert", "message": "Note: prior messages ({hidden} of {total} total messages) have been hidden from view due to conversation memory constraints.\nThe following is a summary of the previous {n} messages:\n {summary}", "time": time})`——格式逐字符照抄（含 `\n {summary}` 的空格）。
7. **trim + prepend**：先 `trim_messages(cutoff)`（= `[messages[0]] + messages[cutoff:]`，保 system——cutoff=0 时全表保留导致 system 重复出现），再 prepend 摘要包 `{"role": "user", ...}`；`messages_total_new = all_time + 1`（摘要也算新增）。

## 样例

```python
msgs = [{"role": "system", "content": "S0"},
        {"role": "user", "content": "x" * 8},
        {"role": "user", "content": "y" * 8},
        {"role": "assistant", "content": "z" * 3}]
led = solve(msgs, 6, 200, len, lambda s: "S<%d>" % len(s), "T")
# 阈值 0.2*200=40：末条 39<40、次末 78>=40 → break，pre_min=2
# min(1,2)=1；messages[1] 是 user → 移位到 2（即便 messages[2] 也是
# user——"ignoring" 注释不生效）→ cutoff=2，seq=[msgs[1]]
```

## 计分

10 个检查点：负 cutoff 与负索引移位、"ignoring" 失效 bug、len<3 退化（-1/-2/IndexError）、递归门恰等（140==140 不触发）、cut2 向零截断与混装串、累计恰等断点、hidden/remaining 记账、trim/prepend 终态（system 重复）、打包格式，以及 150 组随机主上下文对拍。
