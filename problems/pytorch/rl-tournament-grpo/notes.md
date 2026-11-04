# 实现注记（789 rl-tournament-grpo）

## 官方账逐条（reward.py）

- `parse_transcript`：TAG_PATTERN 逐标签扫描；`cursor` 追踪上一标签结束位置——栈空且 tag 外有杂文 → stray。开标签：parse_attrs 校验后入栈；闭标签：栈顶同名 → pop；否则 broken_xml + 回退最近同名（`max(i for ...)`，default=None → continue 忽略），`stack = stack[:index]` 丢弃中间未闭合。栈空时才产块（`if not stack: blocks.append(...)`——顶层块语义）。结尾：栈空有尾杂文 → stray；栈非空 → broken_xml。
- `parse_attrs`：属性间杂文、重复 key → (attrs, False)；call_tool 允许集 {name}、其他标签空集；`valid = 无尾杂文 and 无超集属性`，call_tool 还须有 name。
- `extract_answer`：恰好 1 个 match、strip 非空、内容无 TAG_PATTERN、闭标签后无杂文。
- `check_structure`：blocks[0] 是非空 think；len ≤ 2 → 无工具循环；主循环——每段必须 is_tool_cycle(google_search) 开头，search 循环吃尽后 browse 循环吃尽，回 while。
- `analyze_format`：failures = TAG_ERRORS ∩ errors（固定序）+ tool:unknown_tool_name（扫块）+ answer:invalid_or_missing；**全空才跑结构校验**（官方注释：此时 answer 唯一且是末块，must_end_with_answer / invalid_tool_cycle_order 恒不触发，作者省略）。
- `normalize_scores`：min-max；high==low → 全 0.5。
- `run_tournament`：`rng = random.Random(judge.seed)` 在 4 次重复外创建（**共享**）；每轮 shuffle → `num_groups = len//2` → `candidates[i::num_groups]` 轮转分组（奇数出 3 人组）→ 每组 judge → winner +1 晋级；matches 记 repeat/round/candidates/winner/output。
- `score_group`：formats → answers（截 120000）→ tournament → `reward = reward if answer else 0.0`（**归一后置零**）→ score = tournament + format。

## 沙箱工程

- /tmp/tg_check.py（python3 直跑，纯标准库）：stub data.ResearchExample / judge.JudgeClient；judge 换 (seed, pick) 确定策略（longest=组内最长、first=0 号）。
- **教训**：pick=first 的 3 人组胜者 = shuffle 后组首、随抽签变——不能拍脑袋断 [4,0,0]，直接 dump 官方账（tour3_first raw 也是 [1,3,0]）。

## 判题协议

判题器注入单个 `solve`；`mode` kwarg 分发六入口。`check_*(solve)` 断言集合，6 用例（纯标准库，无 torch 依赖）。
