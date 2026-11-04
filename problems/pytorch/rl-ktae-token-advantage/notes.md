# 实现注记（787 rl-ktae-token-advantage）

## 官方账逐条

### compute_key_tokens.py（信号层）

- `compute_Cohen(a, b)`：两组出现率 clamp [0,1] 后 `asin√a − asin√b`。
- `fisher_exact_test(table)`：对每行 [a,b,c,d]（a=correct 出现、b=wrong 出现、c=correct 未出现、d=wrong 未出现）算超几何精确检验。支撑 k ∈ [max(0, R1+C1−n), min(R1, C1)]，log_comb 用 `lgamma`，p = Σ p_k（p_k ≤ p_obs 且在支撑内）。全向量 151645 长（每 token id 一行）。
- `calculate_ig(table)`：H(Y) − H(Y|X)，log2；`safe_log` 只对 x>0 取 log；条件熵在组总数 >0 时才算；`t_n==0` 的行 IG 置 0。
- `compute_tf`：BM25 饱和。k1=2.0、b=0.5；`avg_len = (correct token 总数 + wrong token 总数)/2`（是 **token 数**不是行数）；`hist/total` 归一后 `(tf·(k1+1))/(tf+k)`。
- `get_distribution`：`torch.unique(return_inverse=True)` 后对每个 token 数 `torch.unique(current_rows)`——**独立行数**。
- `compute_key_tokens`：空 correct 或空 wrong → 全 0 早退；列联表按 `tokenid_set` 遍历、`cc+wc==0` 跳过；融合 `((α·fmt_fisher + β·ig)·x)/(α+β)`（γ_tf 时 x += tf0/(tf1+1e-5) − tf1/(tf0+1e-5)）；`(sigmoid−0.5)·2·top`；**`result[151643] = 0.` 硬编码**。
- `get_key_tokens`：`rewards > 0` 切 correct、`≤ 0` 切 wrong。

### core_algos.py::compute_ktae_outcome_advantage_and_keytokens（L155-236）

- 按 `index` 分组（**numpy 值 hash**）；scores = `token_level_rewards.sum(-1)`。
- 单条组：mean=0、std=1、零权重表 `torch.zeros([responses.max().item()])`——**无 +1**。
- 多条组：`torch.std`（unbiased）+ `ComputeKeyTokens(α=β=γ=1, top=1, max_token_num=responses.max().item())`。
- `scores = (s − mean)/(std + 1e-6)`；`format_weights[i] = id2key_token[index[i]][responses[i]]`（按 **token id** 索引权重表）。
- 最终 `scores.unsqueeze(-1).tile([1, response_length]) * eos_mask + all_weight * eos_mask`——**加法融合**。
- `average_weight_by_mask_tensor` 是死代码（定义未调用）。

## 官方原生 bug（钉死为用例）

1. **小词表 IndexError**：表大小 = max token id + 1 < 151644 时硬编码行越界。
2. **单条组 + 全局最大 token id IndexError**：单条组零表无 +1，行含 responses.max() 必越界。

## 沙箱工程

- /tmp/ktae_check.py（llm env torch 2.10）：stub `transformers`（官方 import 链不需要真模型）+ `sys.modules['verl...compute_key_tokens']` 占位后 exec 官方文件；core_algos 只抽 ktae 函数段（`compute_rloo_outcome_advantage` 行前切）。
- **fixture 语义**：Qwen 词表 id（含 151644 保证表 ≥151644）；`index` 必须传 numpy（tensor 按对象身份 hash 会 KeyError）。
- 长向量 EXPECTED 稀疏化（len + nonzero dict），test 端 round 比较（float32 lgamma/sigmoid 跨 torch 版本低位漂移容忍）。
- **教训**：EXPECTED 必须程序化生成——手抄/手编 151645 长向量的非零位必错（首版 check_fisher/ig/tf/advantage 全因手编数值 FAIL）。

## 判题协议

判题器注入单个 `solve`；`mode` kwarg 分发八入口。`check_*(solve)` 断言集合，12 用例。
