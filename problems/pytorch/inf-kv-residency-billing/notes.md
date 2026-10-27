# 解题说明 · KV 驻留积分与计费

## 积分不是乘法

新手最容易写 `residency = out_tokens * ceil((prompt_len + out_tokens) / block_size)`。这是终值乘法，把整段生成都按「最满」的块数算，结果系统性偏高。正确理解是**逐步累加的黎曼和**：第 k 步的占用是 `ceil((prompt_len + k) / block_size)`，逐步相加。

## 闭式公式：先算阶梯函数的前缀和

如果 `out_tokens` 很大（真实场景几千步），逐步循环也够用；但如果要求 O(1)，可以构造前缀和函数

```
G(T) = Σ_{t=0..T} ceil(t / bs) = bs*K*(K+1)/2 + R*(K+1),  K = T // bs, R = T % bs
```

于是 `residency(p, m) = G(p + m - 1) - G(p - 1)`。这个式子的直觉是：`ceil(t/bs)` 把 t 轴分成若干长度为 `bs` 的台阶，每一级台阶贡献「级号 × 台阶长度」，最后一级不满则按剩余长度结算。

注意 `G(-1) = 0`，这是 `out_tokens = 0` 时区间为空、积分必须为 0 的关键。

## 峰值必须逐时刻累加

峰值不是「每个请求峰值的最大值」，而是**同一时刻所有活跃请求之和的最大值**。实现上要么扫一遍时间轴用字典累加，要么用事件差分（进入 +size、离开 -size）——但块数是变化的，差分不够用，所以最稳的是对每个请求的每一步向 `per_step[t]` 累加。

## 左闭右开区间的两个后果

- 最后一个 token 生成之后立刻释放，所以「占用的步数」恰好等于 `out_tokens`，不额外加一。
- 用 `t = start + k, k ∈ [0, out_tokens)` 覆盖；写成 `[start, start + out_tokens]` 会让峰值与积分都多算一步。

## 计费的两项不可相加的原因

`price_out` 的货币单位是「每 token」，`price_block_step` 是「每块每步」。它们能相加，只是因为业务上定义了汇率；实现时必须分别按各自的量算好再相加，不能把 `out_tokens` 和 `residency` 混在一个量纲里。

## 复杂度

逐步实现是 O(Σ out_tokens)（含峰值），闭式积分是 O(N)；峰值仍需 O(Σ out_tokens)。真实账务系统按时间桶聚合，等价于把 `per_step` 换成直方图。

## 相关题目

- `inf-cache-aware-routing`：把「块」当货币做路由决策。
- `perf-systems` 的 `kv-cache-memory`：静态峰值占用，不含时间积分。
- `inf-paged-kv-block-alloc`：块分配顺序，决定峰值能否被压下来。
