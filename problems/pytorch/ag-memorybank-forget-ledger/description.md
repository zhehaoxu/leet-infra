# MemoryBank 遗忘曲线账

## 题意

MemoryBank（Zhong et al., 2305.10250）给 LLM 伴侣装上基于艾宾浩斯遗忘曲线的长期记忆：记忆随时间衰减、被检索命中就强化。官方实现（zhongwanjun/MemoryBank-SiliconFriend）的 `memory_bank/memory_retrieval/forget_memory.py` 是这条机制的核心。

记忆库、当前日期、检索命中都是外部状态，本题把它们**全部作显式入参**（每条记忆的随机抽样值也显式给出），只考两段账：

- **遗忘账**（`initial_load_forget_and_save`）：按用户插入序 → 日期插入序 → 对话序遍历；旧格式 `[query, response]` 对话迁移写回 dict；缺省 `memory_strength=1`、`last_recall_date=所属日期`、`memory_id=f"{user}_{date}_{i}"` 并 `.update` 写回条目；留存概率 `retention = math.exp(-t / 5*S)`——**官方原文的优先级陷阱：等于 exp(−t·S/5)，强度越大衰减越快**（与论文意图 exp(−t/(5S)) 相反，以官方代码为准）；遗忘判定 `draw > retention`（严格大于，恰等保留）；遗忘下标**逆序 pop**；日期清空时从 history 与 summary **同步删除**；无 `history` 键的用户**整体不进结果库**（官方 continue 位置 quirk）。
- **强化账**（`update_memory_when_searched`）：按 `memory_id.split('_')[1]` 定位日期（缺失即官方 KeyError → 本题 ValueError），首个匹配 `memory_id` 命中后 `strength+1`、`last_recall_date=cur_recall_date` 并 break（重复 id 只改第一个；整日被清空的记忆无法命中）。
- **文档账**：保留对话产出 `时间{date}的对话内容：[|用户|]： {query.strip()}; [|AI伴侣|]： {response.strip()}`（en 版 `Conversation content on {date}:[|User|]: ...`）；摘要不参与遗忘，非 dict 包成 `{"content": ...}` 并写回元数据（`memory_id=f"{user}_{date}_summary"`），产出摘要文档。

## 求解

实现 `solve(bank, now_date, draws, recalled_ids=(), user=None, cur_recall_date=None, language="cn")`，返回 `{"bank", "dialogs", "docs", "updates"}`。域约束：日期为 `YYYY-MM-DD`、draws 为 [0,1] 数值、`language ∈ {cn, en}`；draw 数不足、summary 缺 key、检索日期不存在 → `ValueError`。

## 样例

```python
bank = {"u": {"history": {"2026-01-01": [
    {"query": "q", "response": "r", "memory_strength": 7,
     "last_recall_date": "2025-12-31"}]},
    "summary": {"2026-01-01": "s"}}}
led = solve(bank, "2026-01-01", [0.5])
led["dialogs"][0]["retention"]  # ≈ exp(-1.4) ≈ 0.2466（exp(-t*S/5)）
led["dialogs"][0]["forgotten"]  # True：0.5 > 0.2466
led["docs"]                     # []：日期清空，summary 同步删除
```

## 数据范围

- 用户 ≤ 3、日期 ≤ 9 天窗、每用户对话 ≤ 3 条；200 组随机对拍 + 全锚点精确断言。
