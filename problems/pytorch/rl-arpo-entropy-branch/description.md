# ARPO：熵感知 rollout 分支账

## 题面

多轮工具调用 agent RL 里，环境反馈 token 进入上下文后模型分布的不确定性
会骤升。ARPO（Agentic Reinforced Policy Optimization，arXiv:2507.19849）
把采样预算**在环境反馈后的高熵时刻自适应分叉**：每个 rollout 监测 top-10
截断熵的漂移（相对首见初始熵），熵升的轨迹更容易被分支复制、熵降的更难；
同时保持每样本最终恰好 num_samples 条轨迹。官方 vllm_rollout_with_tools.py
的确定性账（vLLM 引擎/工具执行器/日志剥除，rollout 输出与随机源全入参），
`solve(mode=...)` 五入口：

1. **entropy**：每 rollout 取前 min(20, len(tokens)) 个 token 的 top-10
   logprobs 扁平拼接（缺失用最后一个），熵 = −Σ p·l（p=exp(l)）、空 → 0，
   除 log(vocab_size) 归一化；out_idx 首见才写 initial_entropy_dict
   （此后冻结）。
2. **branch**：每样本 remaining_slots = num_samples − rps（≤0 跳过）；每
   活跃 idx branches_per_idx = min(beam_size − 1, remaining_slots −
   branches_created)；每分支 prob = clamp(rand() − weight×delta, 0, 1)，
   prob > branch_probability 跳过。
3. **refill**：不活跃样本每轮补 1 支、模板取首下标（init_inputs 冷启动）。
4. **round**：工具调用未满 → ("tool", c+1, [])；满额 → ("limit_eos", c,
   [eos])；length 未到上限 → continue 否则 stop；EOS → stop。
5. **reorg**：每样本前 num_samples 个下标、不足复制最后一个补齐。

## 输入/输出

- `entropy`：outputs + active_indices + vocab_size + initial_entropy_dict
  → (current, initial)
- `branch`：active_by_sample + 两熵字典 + rps + num_samples + beam_size +
  branch_probability + entropy_weight + rand_fn → (branches, rps)
- `refill`：active_by_sample + rps + num_samples + sample_to_indices
  → (refills, rps)
- `round`：finish_reason 等 8 参 → (action, new_counter, extra_tokens)
- `reorg`：batch_size + sample_to_indices + num_samples + curr_inputs +
  result_masks + prompt_lens → (sequences, masks)

## 样例

样本 0 两个活跃 rollout [3,4]、num_samples=4、rps=2、beam_size=3：

- idx3 熵 0.9（init 0.5，delta +0.4）：prob = rand − 0.2 → 多数 rand 分支
- idx4 熵 0.5（init 0.7，delta −0.2）：prob = rand + 0.1 → 更难分支
- 2 个名额按活跃序分配、满额 break

## 边界与陷阱

- initial_entropy_dict 冻结：跨回合 delta 累积、首见后不覆盖。
- 三重上限互锁：beam_size=1 时 min(0, …)=0 恒无分支。
- refill 用 init_inputs 冷启动、不是活跃轨迹续写。
- 工具满额 append EOS 终止（不是丢请求）。
- 熵是 top-10 近似 + 20 token 固定窗口——以代码为准，与论文表述有偏差。
