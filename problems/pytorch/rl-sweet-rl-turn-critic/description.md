# SWEET-RL：turn-level critic 账（792）

**来源**：facebookresearch/sweet_rl `scripts/`（rank_best_of_n.py / generate_random_pairs_from_ranks.py / sample_best_of_n.py / evaluate_code.py），论文 arXiv:2503.15478（训练侧在配套 fork YifeiZhou02/collab_openrlhf：multi-turn DPO + length normalization）。

## 题目大意

多轮 agent RL 的 credit assignment 用一个**带 training-time 信息**的 turn-level critic：critic 的 prompt 注入 ground truth（"In light that the final answer is: ..."），对每个 turn 的候选输出打分；policy 用 critic 排序产生的 step-level 偏好对做 length-normalized DPO。模型调用全部入参化——logprob 列表、token 列表、judge 分数、参考基线均为显式入参，只保留脚本里的确定性账。

## 求解什么

`solve(mode=..., ...)` 六入口（mode kwarg 分发，numpy + 标准库）：

1. `mode="judge_prompt"`（turn_input, output, ground_truth=None）→ critic prompt（GT 前缀 + RESPONSE_TEMPLATE 尾巴）
2. `mode="judge_score"`（prompt_logprobs, eos_token, use_sum=False）→ 打分提取账
3. `mode="rank"`（judge_scores, reference_scores, best_of_n）→ 减参考基线后降序 rerank 下标
4. `mode="flatten"`（dialogue_history, ground_truth）→ 逐 turn 记录（input_with_ground_truth 注入）
5. `mode="pairs"`（data, num_pairs=8, seed=42, no_prompt/get_sft/bestworst）→ 偏好对列表
6. `mode="gt_pairs"`（rewards, k）→ 轨迹级 GT 偏好对下标

## 关键官方口径（详见 notes.md）

- judge 四步账：`[:-4]` 截尾（尾部 4 个空响应）→ None 跳过（logprob/token 同步）→ **最后一个 eos** 之后 → sum 或 mean（use_sum = 长度归一开关）
- training-time 信息注入三处：展平 input_with_ground_truth / critic prompt 前缀 / GT 偏好文本前缀——前缀与原文**无空格**
- rank：减同形状参考基线（reshape(-1, best_of_n)）再降序 argsort
- 抽签：升序 argsort **按位置**取——chosen 从前半、rejected 从后半，再按实际分数交换保证 chosen 更高（非绝对最优；bestworst 才取首尾）
- get_sft 官方怪癖：chosen==rejected 都用第一个下标
- gt_pairs：argmax 重复为 chosen、reward **严格 <** max 才出对；下标 = repeat*num_tasks+task
- random.seed(42) 全局种子，choice 顺序与 shuffle 可精确复现

## 计分

判题即账本比对：seed 42 的抽签序列与官方脚本逐对对拍钉死。

## 判题环境

torch + numpy；无模型调用（logprob 全入参）。
