# Tournament-GRPO：淘汰赛奖励账（789）

**来源**：datawhalechina/agentic-rl-lab `experiments/tournament-grpo`（adapted from PuffYang/Tournament_GRPO）。

## 题目大意

LLM-as-judge 直接给全组打分不稳——让同题候选两两淘汰赛：judge 每场只看 2 个答案选一个胜者，胜场数作奖励。轨迹先过 0/1 二值格式门（栈式标签解析 + 结构校验），总分 = min-max 归一锦标赛分 + 格式分。LLM judge 入参化为 (seed, pick) 确定策略、random.Random(seed) 决定抽签，全账确定性。

## 求解什么

`solve(mode=..., ...)` 六入口（mode kwarg 分发，纯标准库）：

1. `mode="parse"`（text）→ [顶层标签块列表, 排序后的错误集]
2. `mode="answer"`（text）→ 严格尾随 `<answer>` 提取
3. `mode="format"`（text）→ {format_reward, failure_reasons, rollout_answer}
4. `mode="normalize"`（scores）→ 组内 min-max（全同分 → 全 0.5）
5. `mode="tournament"`（query, answers, seed, pick）→ {raw_scores, rewards, matches}
6. `mode="score"`（query, solutions, seed, pick）→ {results, log}

## 关键官方口径（回源钉死，详见 notes.md）

- 栈解析：关标签与栈顶不匹配 → 回退最近同名开标签、中间未闭合静默丢弃
- 错误集合去重后按 TAG_ERRORS 固定序输出；只有 call_tool 可带属性且必须 name
- `<answer>` 四重门：唯一、非空、无标签、闭标签后无杂文
- 结构：think 开头非空 → 每段 ≥1 轮 google_search 循环 + 任意轮 browse_webpage → answer
- 锦标赛：M=4 次重复共享 rng、G=2 人/场、轮转分组 `candidates[i::num_groups]`（奇数出 3 人组）、胜者 +1 晋级
- 无答案候选空串参赛、**归一化后**锦标赛分置 0（仍影响他人 min-max 分母）
- 总分 ∈ [0, 2]：锦标赛分 + 格式分

## 计分

- `check_*` 函数集合，EXPECTED 从官方 reward.py 沙箱 dump（/tmp/tg_check.py）
