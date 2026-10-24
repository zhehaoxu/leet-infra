# LLM 推理 prefill/decode 吞吐账

## 题目描述

LLM 推理分两个阶段：**prefill**（并行处理整段 prompt，算力密集）和 **decode**（逐 token 自回归生成，
访存密集）。TTFT（首 token 延迟）与每 token 延迟由此决定。本题按 $2N$ 口径估算两阶段 FLOPs 与耗时。

## 规则

- 实现 `solve(n_params, batch, prompt_len, gen_len, effective_flops)`，返回四元组
  `(prefill_flops, decode_flops, ttft_seconds, total_seconds)`。
- **每 token 前向 FLOPs** 用经典 $2N$ 近似（每个参数贡献一次乘加）：
  $$ \text{flops/token} = 2 \cdot \texttt{n\_params} $$
- **prefill**：一次并行吃下 `prompt_len` 个 token（不随生成长度增长）：
  $$ \text{prefill\_flops} = \texttt{batch} \cdot \texttt{prompt\_len} \cdot 2N $$
- **decode**：串行生成 `gen_len` 个 token，每步只算 1 个新 token：
  $$ \text{decode\_flops} = \texttt{batch} \cdot \texttt{gen\_len} \cdot 2N $$
- **耗时**用给定的有效算力 `effective_flops`（flop/s，已含利用率/MFU 折算）：
  $$ \text{ttft} = \frac{\text{prefill\_flops}}{\text{effective\_flops}}, \qquad
    \text{total} = \frac{\text{prefill\_flops} + \text{decode\_flops}}{\text{effective\_flops}} $$
- `prefill_flops`、`decode_flops` 返回**整数**；`ttft_seconds`、`total_seconds` 返回 **float**。

## 示例

```
输入：n_params=7_000_000_000（7B）, batch=1, prompt_len=512, gen_len=256, effective_flops=1e14
  flops/token = 2 * 7e9 = 1.4e10
  prefill_flops = 1 * 512 * 1.4e10 = 7,168,000,000,000
  decode_flops  = 1 * 256 * 1.4e10 = 3,584,000,000,000
  ttft  = 7.168e12 / 1e14 = 0.07168 秒
  total = (7.168e12 + 3.584e12) / 1e14 = 0.10752 秒
输出：(7168000000000, 3584000000000, 0.07168, 0.10752)
```

## 约束

- `n_params, batch, prompt_len, gen_len` 为非负整数；`effective_flops > 0`
- 返回四元组 `(prefill_flops, decode_flops, ttft_seconds, total_seconds)`
- `total_seconds ≥ ttft_seconds`；两时间项用相对容差 `1e-6` 判定
