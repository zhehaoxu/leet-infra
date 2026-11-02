# notes — ag-memorybank-forget-ledger (order 722)

## 回源口径

- 官方仓 **zhongwanjun/MemoryBank-SiliconFriend**（`zhongwanjun/MemoryBank` 已 404）；遗忘机制在 `memory_bank/memory_retrieval/forget_memory.py`。
- **`forgetting_curve(t, S) = math.exp(-t / 5*S)` 的优先级陷阱**：Python 求值 = exp(((-t)/5)*S) = **exp(−t·S/5)**——强度越大衰减越快，与论文/注释意图 exp(−t/(5S)) 完全相反。以官方代码为准出题，差异记档（这也是本题最强变异锚点：加括号的"论文修复"版本被 curve 锚点抓出）。
- 官方随机遗忘 `if random.random() > retention_probability: forget`——**严格大于**，恰等保留；抽样值显式化作 draws（与遍历序一一对应）。

## 关账点

1. **遍历序**：用户插入序 → 日期插入序 → 对话序；官方迭代的是**原始** user_memory['history']（pop 发生在 deepcopy 上），本题等价实现为 `list(history.items())` 快照——直接在副本上迭代会 RuntimeError。
2. **旧格式迁移写回**：非 dict 对话 `{query: d[0], response: d[1]}` 写回原位。
3. **缺省三元组 + .update 写回**：strength=1、last_recall_date=所属日期、memory_id=f"{user}_{date}_{i}"（memory_id 缺省含下标 i，两处 memory_id-default 变异由锚点抓）。
4. **逆序 pop**：forget_ids.sort(reverse=True) 后按下标 pop（升序 pop 串位——multi_forget 锚点抓）。
5. **日期清空同步删**：history.pop(date) 后 **无条件** summary.pop(date)（官方无 key 检查即 KeyError——本题 ValueError）；随后 summary 分支才有 `if 'summary' in keys` 检查。
6. **无 history 用户整体丢弃**：官方 `if 'history' not in user_memory.keys(): continue` 在 deepcopy **之前**——该用户连 summary 都不进结果库。
7. **摘要永不遗忘**：不消耗 draw；非 dict 包 {'content':...}；metadata 写回（memory_id=f"{user}_{date}_summary"）；summary 文档按剩余日期追加。
8. **强化账**：id.split('_')[1] 定位日期（官方直接下标，日期缺失 KeyError→ValueError）；首个匹配 break；已清空日期的 id 无法命中（除非同日期还有其他记忆）；`+= 1` 与 last_recall_date 刷新。
9. **文档串**：cn `[|用户|]： {q.strip()}; [|AI伴侣|]： {r.strip()}`（**user_kw 后有空格**——f-string 内嵌空格）；en `Conversation content on {date}:`（**冒号后无空格**，q 前空格来自 f'{user_kw} '）；摘要 cn `时间{date}的对话总结为：{content}`。

## 官方不建模

- FAISS/embedding 检索（recalled_ids 显式化）、LLM 日/全局摘要生成、textsplitter 分块、文件 IO、`search_memory` 的日期分组装配（其 source=memory_id 与 page_content 前缀的 date 不匹配，replace 是 no-op——官方自相矛盾处，不入题记档）。

## 出题过程教训

- **官方代码迭代原 dict、pop 在副本**——本题等价实现必须快照 `list(history.items())`，直接迭代副本会 dictionary changed size during iteration（random 首轮即抓出）。
- **forgetting_curve 不可直接从测试调用**（harness 只传 solve）——曲线锚点改从 ledger 的 retention 字段断言。
- 摘要在日期清空时同步 pop → "只剩摘要文档"的直觉断言错误（该日期连摘要一起消失，摘要只对**存活日期**追加）。
- 变异 17 条全 CAUGHT（update-no-break 首轮 PATTERN NOT FOUND 缩进不符，修正后补测 CAUGHT）+ 1 等价（summary 遍历的 list() 快照冗余——循环内只重绑定键值不改大小，SURVIVED 实测确认）。
