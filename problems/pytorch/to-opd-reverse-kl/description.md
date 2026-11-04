# OPD：反向 KL 在线蒸馏账

## 题面

传统蒸馏让学生拟合教师的前向分布（forward KL / JSD）；OPD（On-Policy
Distillation）反过来——**学生先自主采样**，教师对学生实际采出的轨迹逐
token 打 logprob，`reverse KL = student_logprob − teacher_logprob`：学生比
教师更偏好的 token 被惩罚、更不偏好的被奖励，`−reverse_kl` 直接当
advantage 走 importance_sampling 更新。无价值网、无奖励模型、无组基线。
官方 agentic-rl-lab 实验的确定性账（PyTRIO/datasets/swanlab 剥除，rollout
与 teacher 打分全入参），`solve(mode=...)` 四入口：

1. **teacher_logprobs**：compute_logprobs 返回整段 all_ids 的逐 token
   logprob，completion 区间从 `len(prompt_ids)` 切到结尾；长度必须恰等于
   `len(completion_ids)` 且无 None，违反 → ValueError。
2. **datum**：右移四字段账——`prompt_loss_len = len(prompt_ids)−1`；
   input = prompt + completion[:-1]；target = [0]×prompt_loss_len +
   completion；logprobs/advantages 前段 0.0 占位；四序列等长否则
   ValueError；dtype target int64 / 其余 float32。
3. **step**：批账——空序列跳过；`reverse_kl = student_lps − teacher_lps`
   逐 token 向量差；`advantages = −kl_penalty_coef × reverse_kl`；
   reverse_kls 逐 token extend（统计 token 粒度）；datums 全空 →
   RuntimeError；metrics = mean/std（ddof=0）。
4. **schedule**（IDT 交替蒸馏）：build 拼周期元组；resolve 的 steps=0 时
   按两类数据 ceil(size/batch) 步数取 max 周期数乘回周期长度；count 用
   `schedule[step % len]` 计数；batch 用任务自己的 step 下标回绕取批。

## 输入/输出

- `teacher_logprobs`：all_logprobs + prompt_ids + completion_ids → 切片
- `datum`：四入参 → `{input_ids, target_tokens, logprobs, advantages}`
- `step`：batches + teacher_fn + kl_penalty_coef → 四元组
- `schedule`：variant ∈ build/resolve/count/batch

## 样例

prompt 5 tokens + completion 4 tokens：

- teacher 切片：all_lps[5:] = completion 区间
- datum：input 长 8（5+3）、target = [0,0,0,0,201,202,203,204]、
  logprobs/advantages 前 4 位 0.0
- IDT 2:1 周期 7 步 → medical 5 / ceval 2

## 边界与陷阱

- teacher 长度失配走「切片后长度校验」→ ValueError，不 IndexError。
- float32 契约带噪声（−0.6000000238…），测试 round(·,6) 对齐。
- 统计是 token 粒度：空序列跳过不产 datum 也不产统计。
- IDT 的 batch 用任务自己的 step 下标，两类数据互不跳步。
