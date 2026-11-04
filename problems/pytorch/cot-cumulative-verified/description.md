# Cumulative Reasoning：累积态验证（Game24）

## 题意

Cumulative Reasoning（arXiv:2308.04371, 官方仓 `iiis-ai/cumulative-reasoning`）在 Game of 24 上的实现把 proposer、valid 判定、verifier 三类 LLM 调用组织成一个**状态图生长循环**（`Game24/game24-openai.py solve()` L537-590）：每轮从已达状态集合采样一个状态，proposer 给出一步算式与剩余数字串，verifier 给出 sure / likely / impossible 三态结论——likely 才把新状态连同类似父指针、步进与 forbidden 记录入图，sure 则回溯重构完整解题路径。本题把三类 LLM 输出全部显式入参（`rounds` 脚本），考纯账本：

```text
mode="solve":  puzzle + rounds + trycnt -> (success, marker, steps)
mode="state":  同参运行 -> {"records", "parent", "last_step", "forbidden"}
mode="dist":   index + n_states + trycnt -> 采样分布 list[float]
```

## 账本规则（官方逐条钉死）

每轮（下标 i，任何异常**静默跳过该轮**——官方 try/except continue 包住整个循环体）：

```text
tmp = rounds[i].chosen                 # 采样到的已有状态（显式入参）
next_step = next.strip()
a = remaining.strip().split('\n')[0].strip()      # 只取首行
门 A：re.search(r"[^0-9+\-*/.(),=\s]", next_step) -> 跳过
门 B：next_step 或 remaining.strip() 无 \S -> 跳过
judgement == 'Invalid' -> 跳过          # 只有 'Invalid' 拦，'Valid' 等放行
verify == 'sure'：
    steps = [equation.strip().split('\n')[0].strip(),
             next_step + ' (left {a})']
    t = tmp；while t != puzzle：steps.append(last_step[t] + ' (left {t})')；t = parent[t]
    steps.reverse()；返回 (True, i, steps)
verify == 'likely' 且 a 不在 records：
    forbidden[tmp].append(next_step)；forbidden[a] = []
    records.append(a)；parent[a] = tmp；last_step[a] = next_step
verify == 'likely' 但 a 已在 records：整轮静默丢弃（forbidden 也不追加）
其他 verify（impossible 等）：无动作
```

无 success 时返回 `(False, trycnt, [])`——marker 是 **trycnt 本身**而非已跑轮数；rounds 超出 trycnt 的部分被忽略（官方 `for i in range(args.trycnt)`）。

采样分布（`p` 向量构造，官方 L545-553）：

```text
i < 1/2 * trycnt（浮点除，奇数 trycnt 时中点取下整侧）：
    n > 1：全部置 0.5/(n-1) 后把 p[0] 覆盖为 0.5（根偏置，鼓励回到初始状态重开）
    n == 1：p = [1.0]
否则：均匀 1/n
```

## 陷阱

- **重复状态整轮丢弃**：likely 且 a 已在 records 时什么都不发生——forbidden[chosen] 不追加、f/last_step 不覆盖（dup_drop 用例钉 forbidden 只有一个条目）。
- **门 B 的第二个操作数是 remaining 全串 strip**，而 a 是首行——`'6, 8  \njunk'` 的 a 是 `'6, 8'`（首行再 strip 尾随空白），junk 不入状态。
- **last_step 存的是 strip 后完整 next**（可含内部换行），回溯拼 `' (left {t})'` 时换行原样保留。
- **sure 的 steps 结构**：reverse 后最后一项是 equation 首行（合并表达式），倒数第二项才是本轮 next——equation 是 expand LLM 的输出占位。
- **官方成功路径返回的是 expand LLM 的合并输出**；本题无 expand LLM，改为直接返回确定性产物 steps（即喂给 expand 的 `'\n'.join(tmp_steps)` 载荷），在 notes 记录该偏差。
- **静默跳过语义**：chosen 不在 records（官方不可能经 numpy.choice 发生，但本题 chosen 显式入参）→ `forbidden[tmp]` KeyError 被官方同位置的 except 吞掉 → 该轮跳过而非报错。
- 前半程判定用**浮点除** `i < 1/2 * trycnt`：trycnt=5 时 i=2 仍偏置、i=3 才均匀。

## 判题环境

无 numpy / guidance / openai 依赖——采样（choice）、重试包装（run 的 MAX_RETRY）、expand 合并与 sympy 验算均不入题；p 向量用纯 python float 构造（与 numpy fill 同值）。
