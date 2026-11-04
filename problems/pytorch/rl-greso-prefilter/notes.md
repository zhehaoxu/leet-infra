# 实现注记（788 rl-greso-prefilter）

## 官方账逐条

### presampling_selector.py::DataProfiler（skip 族）

- `add_reward`：`data[f"{source}_{index}"].append((epoch, reward))`。
- `easy_example_skip(current_epoch, data_id, n=-1)`：trace 内任意 `(epoch, reward==1)` 且 `epoch + n ≥ current`；n=−1 → 10e7（永久）。
- `easy_linear_backoff_skip(epoch, data_id, k)`：从尾数连续 `reward ≥ 0.98`（<0.98 断）→ count；count==0 早退；`backoff = randint(0, k·count)`；`max_epoch + backoff ≥ current` → True。max_epoch 取**全 trace**。
- `hard_linear_backoff_skip`：门——**最新** reward >0.11 → False（可解例不 hard-skip）；从尾连续 `≤ 0.11` count（>0.11 断）；其余同 easy。
- `easy/hard_probabilistic_skip(p)`：同 count（无 count==0 早退）；`skip_prob = 1 − max(p^count, 0.01)`；`torch.rand(1) < skip_prob`。**rand 惰性**——门/count 短路时不采样。
- `filter_examples_all_probabilistic(epoch, batch, p_easy, p_hard, return_log)`：先 easy 后 hard；skip_log `[data_id, 0|1|2]`（0=easy、1=hard、2=保留）。

### ttis.py::filter_zero_advantage

- reward reshape (−1, n)；`max(dim=1) == min(dim=1)` → 零方差组；保留组展开 `idx·n + arange(n)`。

### ray_trainer.py 尾段（调度账）

- `targeted_easy = target_zero_variance/3`、`targeted_hard = ×2/3`（hard 侧 2/3 探索余量）。
- 预过滤累积：`acc_batch = concat(acc_batch, filtered)`；`len(acc_batch) ≥ adaptive_sampling_bs` 才 slice（n_gpus 整除）放行，否则 continue。
- refill：`estimated = int(reminder / (1 − zvr) / n * 1.25)`；`estimated ≤ default → last_sampling=True`；`adaptive = clamp(estimated, 64, default)`；`full_sampling_bs` → adaptive=default、last=False。
- p 自适应（epoch > 0）：`hard_ratio ≥ targeted_hard → p_hard = max(p−0.01, min_p)` 否则 `min(p+0.01, max_p)`；easy 对 targeted_easy 同理。

## 沙箱工程

- /tmp/greso_check.py（python3 直跑，纯标准库）：DataProfiler 整类直通、randint/rand 换 stub 序列（`lambda a, b: b − 1` 等确定值）。
- **教训：rand 惰性语义**——首版 solution 在 filter 层 `pop(0)` 提前消费 rand，与官方「门槛短路不采样」不符 → check_filter IndexError/错位。入参化随机源必须用 callable（`rand_fn()` 只在真正走到随机比较时调用）。
- dump 元组被 json 序列化为列表——test 端 `[list(t) for t in ...]` 规范化比较。

## 判题协议

判题器注入单个 `solve`；`mode` kwarg 分发七入口。`check_*(solve)` 断言集合，9 用例（纯标准库，无 torch 依赖——随机源全入参）。
