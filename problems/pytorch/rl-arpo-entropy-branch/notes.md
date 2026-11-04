# ARPO 题解笔记

## 官方账的关键细节（沙箱实测钉死）

1. **熵是 top-10 截断分布的 −Σ p·l**：`self.logprobs = 10` 固定、
   `_calc_entropy` 对拼接后的 top-10 logprobs 一次求和——不是全词表熵、
   也不是逐 token 熵再平均。归一化 `math.log(vocab_size)`。
2. **20 token 窗口 + 缺失回退**：`min(20, len(tokens))` 固定窗口；某 token
   的 logprobs 取不到用**最后一个**（`except: logprobs[-1]`）——vLLM 首
   token 边界的怪癖。
3. **initial_entropy_dict 冻结**：`if out_idx not in initial_entropy_dict:
   initial_entropy_dict[out_idx] = entropy`——首见写入后不再更新。分支
   门控的 delta 是对首次观测的漂移，跨回合累积。
4. **prob 方向**：`prob = random.random() − entropy_weight × delta`、clamp
   [0,1]、`prob > branch_probability → continue`。熵升（delta>0）→ prob
   降 → 更易分支；熵降 → prob 升 → 更难。rand() 是外层随机源、delta 是
   确定性调制——同一 rollout 的多次分支尝试共享同一 delta。
5. **三重上限互锁**：remaining_slots（num_samples − rps）→ 总量封顶；
   branches_per_idx = min(beam_size − 1, remaining_slots − created) → 单
   idx 封顶；branches_created 达 remaining_slots 即 break。beam_size=1 时
   min(0, …)=0 恒无分支。
6. **refill 每轮 1 支 + 冷启动**：不活跃样本用 `sample_to_indices[orig][0]`
   的 **init_inputs** 模板重开（curr=init、masks 空、counter 0）——不是
   从活跃轨迹续写。
7. **工具满额 append EOS**：`curr_inputs[out_idx].append(eos_token_id)` +
   result_masks 补 1 + call_limit_reached 计数——不是丢请求。
8. **输出重组的复制补齐**：`while len(selected) < num_samples: append(
   selected[-1])`——分支预算的下限兜底；response 从 prompt_len 切。
9. **与论文的偏差**：论文讲"环境反馈 token 的后验熵"，官方实现是 top-10
   近似 + 固定 20 token 窗口 + 对首见熵的漂移——以代码为准。

## 与相邻题的划界

- `to-grpo-advantage`：GRPO 的组内归一化在 advantage 侧；ARPO 的预算分配
  在 rollout 侧（样本数动态向高熵分支倾斜），本题只收 rollout 侧账。
- `to-dapo-loss`：DAPO 的 dynamic sampling 是 batch 内全对/全错组重采；
  ARPO 是同一 prompt 内的轨迹分叉、发生在生成阶段。

## 沙箱搭建教训

- 熵监测段的 initial 记录在 `current_entropy_dict[out_idx] = entropy`
  **之后**——end 锚用 `self.initial_entropy_dict[out_idx] = entropy` 行，
  别用 current 赋值行（会切掉 initial 冻结语义）。
- 段首锚 `entropy_norm_factor = math.log(vocab_size)` 行本身要保留（归一
  化因子），替换的是它上一行的 get_vocab 调用。
- json dump 把元组列表转成列表、int 键转字符串——test 端比较要
  `[list(b) for b in branches]` 与 `{str(k): v for k, v in rps.items()}`
  规范化。
- round 的三元组返回同理 list() 规范化。
