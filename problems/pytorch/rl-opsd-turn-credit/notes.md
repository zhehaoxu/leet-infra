# AgentOPSD 实现笔记（790）

回源：`datawhalechina/agentic-rl-lab` `experiments/agentopsd/`（advantages.py 196 行纯 numpy + teacher.py::_score_turns + loss.py::build_agentopsd_ppo_datum）。论文 arXiv:2608.05987。官方 SkillBank 来源 pinned 到 `ZethWang/AgentOPSD@0c478b2d`（skills.py 注释），本题不需要——技能文本已并入 teacher 前缀长度差。

## 逐入口口径

### group_sequence_advantages（官方 advantages.py::60-73）

- `mean_reward = sum(rewards)/len(rewards)`、`std = np.std(rewards, ddof=0)`、`adv = (r−mean)/(std+ε)`。
- **返回 (advantages, mean_reward)**——mean_reward 是未 clip 的组均值，作 initial belief 传给 reshape。ε 由调用方传（assign 传 config.epsilon）。
- 全同分组：std=0 → adv 全 0；单条轨迹组同理。这两个退化形态是 degenerate 判定和 ppo 过滤的源头。

### _sigmoid（官方 76-83）

数值稳定两分支：`value >= 0` 走 `1/(1+exp(−v))`，否则走 `exp(v)/(1+exp(v))`。分支边界（v=0）两侧结果一致。

### reshape_turn_advantages（官方 86-156，核心）

逐字段对应 TurnCredit dataclass，顺序钉死：

1. `b0 = min(max(initial_belief, ε), 1−ε)`（**initial_belief 是未 clip 的组均值**）；`logit_b0 = log(b0/(1−b0))`。
2. 递推（c_0=0）：`cumulative = γ·cumulative + evidence`；`belief_after = σ(logit_b0 + cumulative)`；`raw.append((evidence, cumulative, belief_before, belief_after))`；`belief_before = belief_after` 链式推进。
3. `sign = +1 / −1 / 0`（按 sequence_advantage 的符号，0 显式分支）。
4. `outcome_credit_k = sign · (after − before)`——**用 raw 里的 before/after 差**，不是 belief_after 与 b0 的差。
5. `z = (credit − mean(credits)) / (std(credits, ddof=0) + ε)`；单 turn 时 std=0、credit 自身=0 → z=0。
6. `weight = min(max(1 + b·z, 1−b), 1+b)`（默认 [0.8, 1.2]）；`multiplier = (1−λ) + λ·weight`（默认 [0.9, 1.1]）；`A_turn = A_seq · multiplier`。
7. turn_index 从 1 起；`delta_belief = after − before`（与 outcome_credit 差一个 sign 因子）。

**零优势轨迹**：sign=0 → credit 全 0（`0.0` 与 `−0.0` 都会出现，测试用容差比较）→ z 全 0 → weight 全 1 → multiplier=1 → A_turn 全 0。evidence/belief 账仍照算。

**clip 边界**：initial_belief=1.0 → b0=0.9999，evidence=0 时 after 与 before 只差 1e-16 级浮点噪声；initial_belief=0.0 → b0=1e-4，σ(logit(1e-4)) 往返出 9.99999e-05 与 −1.35e-20 噪声。测试全部用 1e-9 容差。

### _score_turns（官方 teacher.py::63-87）

- `prefix_offset = teacher_initial_length − student_initial_length`。teacher 前缀含技能提示更长；**全部 action 区间平移 +offset**（teacher_start = action_start + offset）。
- 每 turn：`numeric = [float(v) for v in teacher_logprobs[start:end]]`；`evidence = sum(t − s for t, s in zip(numeric, student_logprobs))`。
- **zip 截断**：teacher 切片与 student logprobs 不等长时按短侧求和。官方真实数据二者等长（action 区间就是 student 生成的 token），截断是隐式行为——fixture 用 (action_start=6, action_end=10, 2 个 lp) 显式钉住。
- action_tokens 累加 `len(numeric)`（teacher 侧长度，非 zip 后长度）。
- 官方还把 numeric/float(evidence) 写回 step.teacher_logprobs/step.teacher_evidence——本题以返回值 (evidences, action_tokens, slices) 替代。

### assign_agentopsd_advantages（官方 159-196）

- `groups: dict[str, list]`，key 是 `str(group_id)`（`defaultdict` 在官方实现，语义同 setdefault）。
- 每组先算序列优势与 initial belief，**退化判定在写回之前**：`all(abs(adv) <= ε)`。
- 每条轨迹 `reshape_turn_advantages(adv, initial_belief, turn_evidences)` 后写回：`step.teacher_evidence = turn.evidence`、`step.turn_advantage = turn.advantage`，按 `zip(trajectory.steps, credit.turns)` 截断——**steps 少于 evidences 时只填前缀**。
- 返回 AdvantageStats(groups, degenerate_groups)。本题以 (results, groups, degenerate_groups) 返回，results 逐轨迹 dict（group_id/sequence_advantage/initial_belief/turn_advantages/step_evidences/step_turn_advantages），**结果顺序与输入顺序一致**（官方按对象属性写回天然保序；本题用 id 映射保序——不能用 list.index()，相同 evidences 的两条轨迹会错位）。

### build_ppo_arrays（官方 loss.py::42-77）

- 全 token 先填 0.0（old_logprobs_by_token / advantages_by_token 两个数组）。
- `zip(steps, turn_credits)`：position = `action_start + offset`（**不需要 action_end**，官方 loss.py 只用 action_start + enumerate(logprobs)），把 old_logprob 与该 turn 的 credit.advantage 广播进去；action_tokens 累加 len(logprobs)。
- 右移对齐：`input_tokens = full[:-1]`、`target_tokens = full[1:]`、`old_logprobs = old_by_token[1:]`、`advantages = adv_by_token[1:]`——position p 的标签是 token p+1。**最后一个 action token 的 logprob 落在 [:-1] 的末位不被丢弃**（advantages_by_token 与全 token 等长，取 [1:] 后仍覆盖）。
- zip 截断：steps 多于 turn_advantages 时多余的 step 不写。
- 官方 `build_training_datums` 过滤 `sequence_advantage != 0.0`——退化组整组不进训练。测试在 assign 用例里用 python 表达式钉这个语义。

## 沙箱与测试

- 沙箱脚本 `/tmp/opsd_check.py`：官方 advantages.py 直接 sys.path import（纯 numpy 零依赖）+ `_score_turns` 逐字符复刻 + ppo 数组复刻；dump `/tmp/opsd_expected.json` 与自定义 config 的 `/tmp/opsd_expected_cfg.json`（γ=0.5, λ=0.8, b=0.5, ε=0.01）。EXPECTED 全部程序化生成写进 test_cases.py（16 组）。
- 对拍：solution 五入口 × 全 fixture 与 dump 逐字段 1e-9 容差全等。
- 覆盖矩阵：三组 GRPO（正常/全同退化/单条）· 正/负/零优势轨迹全字段 · clip 上下界 · 自定义 config 四 turn + 单 turn · prefix_offset 三 turn 切片 + 手工核对 · zip 截断（score 与 ppo 双侧）· assign 分组/退化统计/截断写回/训练过滤 · ppo 右移对齐与 0 填充。

## 变异清单（mut790 待做）

见 test_cases.py 的 7 个 check 函数；易漏点：ddof=0 vs ddof=1、clip 的 min/max 次序、c_0 是否为 0（首轮 cumulative）、sign(0) 分支、weight 的 clip 边界、multiplier 的 (1−λ) 基座、prefix_offset 的方向、zip 截断、右移取 [1:]、退化判定 <= vs <。

## 划界

- `to-grpo-advantage`：那题是 advantage 侧组内标准化本身；本题标准化之后还有信念递推 + 边际信用 + bounded multiplier 三层重塑，且 credit 粒度从序列降到 turn。
- 784 `rl-vineppo-mc-value`：VinePPO 用 MC 回报估计替代价值网（每步 k 条续采）；本题不采样，只做一次 teacher forward 的 logprob 差账。
- 786 `rl-arpo-entropy-branch`：ARPO 是 rollout 时按熵分支扩展；本题是 rollout 后的优势重塑。
