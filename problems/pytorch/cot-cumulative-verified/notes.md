# 解题笔记

## 思路

官方 iiis-ai/cumulative-reasoning 的确定性核心：

1. **Game24/game24-openai.py solve()**（L537-590）：状态图生长循环——records/last_step/f/forbidden 四件套记账、两阶段采样分布（L545-553）、四道门（字符集、非空白、Invalid 判定）、verifier 三态分流（L562-585）、父链回溯重构（L569-574）；
2. **is_sure / valid_output / valid_judgement**（L38-41、L135）：`is_sure = output == 'sure'`、三态 `["sure", "likely", "impossible"]`、判定 `["Valid", "Invalid"]`。

回源说明：todos ⑤ 原描述「proposer+verifier 累积态账——命题经 verifier 准入、累积态单调增长」与官方 Game24 实现吻合（FOLIO 版是五道门准入流水线、MATH 版是提示词流水线无逐命题验证，二者机制增量薄不立项）。

上游不入判题域：guidance 程序与 prompt 组装、OpenAI 调用与 run() 的 MAX_RETRY 重试、numpy.random.choice 采样（chosen 显式入参）、expand 合并 LLM（其输出为显式入参/回溯载荷直接返回）、`__main__` 的 sympy 验算、tqdm/pandas。

与现有题划界：
- **rs-toT / rs-lats / toT 系**：树搜索节点由打分函数筛选；本题状态由 **verifier 三态语义**准入（likely 准入 / sure 终止 / impossible 丢弃）且路径按父指针回溯；
- **775/776/777**：demo 拼装、选题、候选评分；本题是**跨轮累积状态图**——状态是派生串（剩余数字组合），不是文本候选；
- **778**：RoT 的子问题序列由题目语法机械派生且一次性编译；本题每轮 proposer 输出显式入参、verdict 驱动分叉。

## 关键实现点（官方口径逐条钉死）

- **轮内处理顺序**：chosen → a=首行抽取 → 门 A（`[^0-9+\-*/.(),=\s]` 之外字符）→ 门 B（`\S` 检查，第二操作数是 remaining **全串** strip）→ Invalid → verifier。门 A/B/Invalid 都是纯 continue（互换等价）。
- **a 的两次 strip**：先全串 strip 再取 split('\n')[0] 再 strip——`'6, 8  \njunk'` → `'6, 8  \njunk'` → 首行 `'6, 8  '` → `'6, 8'`；junk 既不进状态也不进门 B。
- **likely 准入五件套**：`forbidden[tmp].append(next_step)`（先）→ `forbidden[a] = []` → `records.append(a)` → `parent[a] = tmp` → `last_step[a] = next_step`。删掉 `forbidden[a] = []` 不会立刻暴露——要等 a **被选为 chosen** 时 `forbidden[a]` KeyError 才跳过后续轮（链式准入用例）。
- **sure 的 steps 结构**：`[equation 首行, next + ' (left a)']` 起底，while 沿 parent 向根走、每步 `last_step[t] + ' (left t)'`，最后 reverse——根侧步在前、equation 合并式在最后。official 返回 expand LLM 输出，本题返回 steps（喂给 expand 的 join 载荷），偏差已记录。
- **重复状态静默丢弃**：likely 且 a in records 时**什么都不做**——forbidden[tmp] 不追加（dup_drop 用例钉 forbidden 只有一条）。
- **失败 marker = trycnt**：不是已跑轮数（rounds 不足 trycnt 提前收场也返回 trycnt）。
- **两阶段采样分布**：`i < 1/2 * trycnt` 是**浮点除**——trycnt=5 时 i=2 仍偏置；n>1 时先 fill(0.5/(n-1)) 再覆盖 p[0]=0.5，n=1 直接 p[0]=1.0。
- **静默跳过**：官方 try/except 包住整个循环体，任何异常（含 `forbidden[tmp]` KeyError）→ continue。本题 chosen 显式入参，chosen 不在 records 的轮按同一语义跳过。

## 测试设计（11 例）

- `dist`：偶/奇 trycnt 的前半程边界（i=2 偏置、i=3 均匀）、n=1、trycnt=0；
- `success_basic`：Invalid 门 + 填充式 strip 的 next（存储已 strip）+ sure 重构三步 + state 快照；
- `deep_chain`：两层祖先回溯（根侧步在前、equation 收尾）；
- `gates_no_success`：门 A（'!'）/ 门 B（全空白）/ 正常准入 ×2 / 重复丢弃 / impossible——state 钉 records/parent/last_step/forbidden 全量；
- `dup_drop`：重复 likely 整轮丢弃、forbidden 单条目；
- `multiline_edges`：remaining 首行抽取（尾随空白二次 strip）、next 内部换行入 last_step、equation 尾随空白被首行二次 strip 剪掉；
- `forbidden_accumulation`：同一父状态两次准入 → forbidden 两条目 + impossible 轮无动作 + 从孙状态回溯；
- `success_early`：首个 sure 立即返回，后续 sure 不覆盖 marker；
- `round_boundaries`：rounds>trycnt 截断 / rounds<trycnt 提前收场（marker=trycnt）/ chosen 不在 records 静默跳过 / trycnt=0；
- `purity`；`args_validation`：mode/puzzle/trycnt/rounds/轮键/dist 参数 + bool 排除 + 轮内坏值（next=None）静默跳过。

## 变异收口

44 条设计 → **41 条 CAUGHT + 3 条等价剔除**（m-gateA-B-order：两道门纯 continue 互换无副作用；m-likely-a-strip：官方 `a = a.strip()` 死代码加回等价；m-eq-nostrip-head：fixture 无前导空白 equation，首轮 strip 冗余）。

覆盖：dist 五态（浮点/整除中点、根偏置、fill 值、均匀值、n=1）、门四态（门 A 删除/字符集收窄、门 B 删除/or→and）、judgment 两态（大小写、Valid 拦截反转）、verifier 三态（sure 反转、likely 大小写）、准入五态（重复检查双向、parent 指根、last_step 换值、forbidden 追加删除/新建删除）、sure 路径七态（equation 整串/无尾 strip/丢弃、left 占位错用、祖先 left 错用、不 reverse、回溯跳根、while 恒真）、抽取三态（remaining 不分首行/无尾 strip、next 不 strip）、循环三态（上界换 len(rounds)、marker 换 len(rounds)、sure 不 break）、校验八态。

## 工程学备注

- **官方沙箱对拍先行**：把官方 solve() 源码 exec 进 stub 环境（run/choice/args/pbar 四件 stub），全部 fixture 期望（含 state 快照与 steps 载荷）由沙箱 dump——手推 4 层状态图必错。stub 的关键约定：**gen 调用读 rounds[idx] 并推进 idx，valid/verifier/expand 读 rounds[idx-1]**（每轮一个 dict，verdict 与 chosen 同属一轮）。
- **官方成功路径返回 expand 输出而非 steps**——直接照抄会让回溯重构在判题域里不可观察（mutation 全部等价）；改返回 steps 载荷后 m-parent-walk/m-steps-noreverse 等 7 条变异才可捕获。
- **state 模式的必要性**：m-gateA-drop、m-dup-check-true、m-forbidden-noappend、m-bound-len 四条变异在 solve 模式下结果恒同（False, N, []），只有账本快照能区分——观测窗口是变异收口的刚需。
- **变异脚本的字符类陷阱**：往 negated class 里**加**重复字符（'=' 已在类中）是恒等变异——收窄语义要**移除**字符。
