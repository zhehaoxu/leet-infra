# GRESO：rollout 前选择性预过滤账（788）

**来源**：arXiv:2506.02177（Act Only When It Pays / GRESO），official `Infini-AI-Lab/GRESO`（verl fork）。

## 题目大意

用每条 prompt 的历史 reward trace 预测其 rollout 价值：一直全对（太简单）或一直全错（不可解）的 prompt 在 **rollout 之前**被跳过；省下的预算由零方差过滤后的自适应补采样吃满；easy/hard 的过滤概率还按实测比例做跨步自适应。全部确定性账：trace、batch、随机源全为显式入参。

## 求解什么

`solve(mode=..., ...)` 七入口（mode kwarg 分发，纯标准库）：

1. `mode="id"`（data_sources, indices）→ `["{source}_{index}", ...]`
2. `mode="trace"`（ops, data_id）→ 追加式 (epoch, reward) 轨迹
3. `mode="skip"`（ops, current_epoch, data_id, variant, ...）→ 单例 skip 判定（五变体）
4. `mode="filter"`（ops, current_epoch, batch, p_easy, p_hard, rand_values）→ (selected, skip_log, counts)
5. `mode="refill"`（reminder, zero_variance_ratio, n, default_bs, ...）→ (bs, last_sampling, used_full)
6. `mode="adapt"`（epoch, hard_ratio, easy_ratio, p_hard, p_easy, ...）→ (p_hard, p_easy)
7. `mode="zero_variance"`（rewards, n）→ (组 flags, 保留的扁平下标)

## 关键官方口径（回源钉死，详见 notes.md）

- easy = 从尾连续 reward ≥0.98；hard = 最新 reward ≤0.11（可解例保护门在前）+ 从尾连续 ≤0.11
- backoff = randint(0, k·count) 锚在**全 trace** max_epoch；probabilistic 的 skip_prob = 1 − max(p^count, 0.01)，count==0 不短路
- **rand 惰性采样**：门槛短路的检查不消耗随机源
- refill 估算 = int(reminder/(1−zvr)/n·1.25)，clamp [64, default]，estimated≤default 置 last_sampling
- p 自适应：ratio ≥ target（hard=target·2/3、easy=target/3）→ p −0.01（滤更多）否则 +0.01，clamp [0.05, 0.95]，epoch 0 冻结
- 零方差过滤：组内 reward max==min 整组丢弃

## 计分

- `check_*` 函数集合，EXPECTED 从官方代码沙箱 dump（/tmp/greso_check.py，随机源入参化）
