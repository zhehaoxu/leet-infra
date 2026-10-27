# KV 驻留时长积分与 token 计费账

## 问题

推理服务的账单位于两个完全不同的维度上：

- **按 token 计**：`output token` 数 × 单价，这是对外报价的主口径。
- **按驻留计**：每块 KV 从被分配到被释放，一共占了多少「块·步」。这是对内算成本、算扩容、算 batch 上限的口径，也是判断「长尾请求到底贵在哪」的唯一办法。

难点不在计价公式，而在**驻留量随生成过程变化**：请求开始时只占 `ceil(prompt_len / block_size)` 块，每多生成 `block_size` 个 token 才多占一块。所以「块·步」是一个关于时间步的求和，不是「块数 × 步数」。

## 任务

实现 `solve(prompt_len, start, out_tokens, block_size, price_out, price_block_step)`：

- 输入：`prompt_len` `(N,)`、`start` `(N,)`（开始占块的步）、`out_tokens` `(N,)`、`block_size`、两个单价（均 `>= 0`）。
- 请求 r 在全局步 `t ∈ [start_r, start_r + out_tokens_r)` 内持有
  `blocks(r, t) = ceil((prompt_len_r + (t - start_r)) / block_size)` 个块，退出时立即释放。
- 输出：
  - `residency` `(N,)` int64：每请求的块·步积分；
  - `charge` `(N,)` float32：`price_out * out_tokens + price_block_step * residency`；
  - `peak_blocks` int64 标量：全时间轴上同时占块数的峰值；
  - `total_charge` float32 标量。

## 示例

```
prompt_len = [4, 10], start = [0, 1], out_tokens = [3, 2]
block_size = 4, price_out = 0.5, price_block_step = 0.1

r0: t=0 -> ceil(4/4)=1；t=1 -> ceil(5/4)=2；t=2 -> ceil(6/4)=2   积分 = 5
r1: t=1 -> ceil(10/4)=3；t=2 -> ceil(11/4)=3                     积分 = 6

时间轴占用: t=0 -> 1, t=1 -> 2+3=5, t=2 -> 2+3=5, t=3 -> 0        峰值 = 5

charge(r0) = 0.5*3 + 0.1*5 = 2.0
charge(r1) = 0.5*2 + 0.1*6 = 1.6
total_charge = 3.6
```

## 约束

- **驻留量随时间增长，不能用终值代替**：`out_tokens × ceil((prompt_len + out_tokens) / block_size)` 会明显高估，因为前期占的块少。
- **区间是左闭右开**：`[start, start + out_tokens)`，最后一次生成之后不再占块，`out_tokens = 0` 的请求积分为 0。
- **`prompt_len` 正好是 `block_size` 的整数倍时**，生成第一个 token 就要多占一块——这是最容易差一的地方。
- **峰值是「同时占块数」**，不是单个请求的最大值；多个请求重叠时它会叠加。
- **`block_size` 与两个单价必须为正/非负**，`out_tokens` 可以为 0。
