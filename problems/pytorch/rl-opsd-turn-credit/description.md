# AgentOPSD：turn-level credit 账（790）

**来源**：datawhalechina/agentic-rl-lab `experiments/agentopsd`（adapted from ZethWang/AgentOPSD，arXiv:2608.05987，清华/浙大/美团）。

## 题目大意

GRPO 把序列级 advantage 广播到轨迹里所有 action token——长程 ALFWorld 轨迹里前面的绕路和最后的关键动作一起被鼓励。AgentOPSD 的解法是加一层特权自蒸馏：Self-Teacher（同一参数快照 + 看得到 SkillBank 技能）沿 Student 已执行的**同一条轨迹**重新打分，token 级 logprob gap 沿 turn（两次环境观测之间的一段动作）求和成 evidence；但**孤立的局部 gap 不直接当 credit**——把 evidence 递推进一个受贝叶斯启发的累积信念，credit 取信念的边际变化 ΔB，再经 bounded multiplier 重塑序列优势。pytrio/Trajectory 依赖全部入参化：步级 logprob、action token 区间、终局奖励均为显式入参，全账确定性。

## 求解什么

`solve(mode=..., ...)` 五入口（mode kwarg 分发，numpy + 标准库）：

1. `mode="group"`（rewards, epsilon）→ (组内 GRPO 序列优势, 未 clip 的组均值奖励作 initial belief)
2. `mode="reshape"`（sequence_advantage, initial_belief, evidences[, γ/λ/b/ε]）→ (clip 后 initial belief, 每 turn 十字段账)
3. `mode="assign"`（trajectories: (group_id, reward, turn_evidences, n_steps) 元组[, config]）→ (逐轨迹结果, 组数, 退化组数)
4. `mode="score"`（steps: (action_start, action_end, logprobs) 元组, teacher_logprobs, student/teacher 前缀长）→ (每 turn evidence, action token 数, teacher 切片)
5. `mode="ppo"`（full_tokens, steps: (action_start, logprobs) 元组, turn_advantages）→ {input_tokens, target_tokens, old_logprobs, advantages, action_tokens}

## 关键官方口径（回源钉死，详见 notes.md）

- initial belief 是**未 clip 的组均值奖励**；clip 只发生在 reshape 内部 [ε, 1−ε]
- 证据递推 c_k = γ·c_{k-1} + e_k 从 c_0=0 起，belief_before 链式推进（不逐 turn 重置）
- **outcome_credit = sign(A_seq)·(B_k − B_{k-1})**：sign(0)=0 → 零优势轨迹 credit 全 0、weight 全 1
- weight 三段账：z 标准化 → w = clip(1+b·z, 1−b, 1+b) → multiplier = (1−λ)+λ·w；**只重分配幅度、不反转方向**
- score：prefix_offset = teacher 前缀长 − student 前缀长，全部 action 区间平移；zip 截断语义
- ppo：非 action token 全 0，右移对齐（input=tokens[:-1]、target=tokens[1:]、logprobs/advantages 取 [1:]）
- sequence_advantage == 0 的轨迹（退化组）不进训练

## 计分

本题无"得分"概念，判题即账本比对：全部返回值与官方 advantages.py/teacher.py/loss.py 的字段一一对应，测试 EXPECTED 由官方源码沙箱 dump 生成。

## 判题环境

torch + numpy，无 pytrio / transformers；无随机性（全确定性账）。
