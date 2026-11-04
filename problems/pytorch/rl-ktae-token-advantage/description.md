# KTAE：token 级优势估计账（787）

**来源**：arXiv:2505.16826（KTAE），official `ZNLP/KTAE`（verl fork）。

## 题目大意

Outcome reward 只给整条序列打分。KTAE 的做法是：把 batch 内按奖励符号切成 correct（>0）/wrong（≤0）两组 rollout，统计每个 token id 的 2×2 列联表（组内出现行数 / 对组出现行数 / 各自未出现行数），用四个统计信号——Fisher 精确检验、信息增益、BM25 饱和词频、arcsine-Cohen 距离——合成一个「key token 权重表」；advantage = 组内标准化回报（GRPO 风格）沿 eos mask 广播 **加上** 按 token id 索引的权重。全部确定性账：rollout、奖励、分组下标全为显式入参。

## 求解什么

`solve(mode=..., ...)` 八入口（mode kwarg 分发，torch float32）：

1. `mode="distribution"`（tensor, mask）→ `{token_id: 出现的独立行数}`
2. `mode="appear_table"`（responses_ids, mask, rewards）→ 每行 `[cc, wc, tc−cc, tw−wc]` 的列联表张量
3. `mode="fisher"`（table）→ 格式化后的 Fisher 信号 `exp(−2p)`，p==1 → 0
4. `mode="ig"`（table）→ 信息增益向量（log2）
5. `mode="tf"`（correct, correct_mask, wrong, wrong_mask, table）→ BM25 饱和词频 [N+1, 2]
6. `mode="cohen"`（table）→ arcsine−sqrt 距离向量
7. `mode="key_tokens"`（responses_ids, mask, rewards[, alpha, beta_ig, gamma_tf, top, max_token_num]）→ 权重表
8. `mode="advantage"`（token_level_rewards, responses, eos_mask, index[, epsilon]）→ [bsz, response_length]

## 关键官方口径（回源钉死，详见 notes.md）

- 表大小 = `max_token_num + 1`（调用方传 `responses.max().item()`），硬编码 `result[151643] = 0`
- 单条组零权重表大小 = `responses.max()`（**无 +1**，与多行组不一致——原生 bug）
- `get_distribution` 数独立行数（unique rows），不是频次
- Fisher p 值是 lgamma 数值逼近 + 支撑掩码累加；p==1 压 0
- key token 符号由 Cohen 差决定：correct 独占 → sigmoid 饱和 +1、wrong 独占 → −1、频率同 → 0
- advantage 融合是**加法**（标准化 tile + 权重，各乘 eos_mask）；空 correct/wrong 组 → 权重全 0
- `index` 按值 hash（官方传 numpy）；`torch.std` unbiased

## 计分

- `check_*` 函数集合，EXPECTED 从官方 torch 代码 dump（沙箱 /tmp/ktae_check.py，llm env）
- 长向量（~151645）稀疏存储，数值 round 比较（float32 低位差容忍）
