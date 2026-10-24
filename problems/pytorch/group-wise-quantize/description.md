# 分组量化（Group-wise Fake-Quant）

## 题目描述

LLM 权重 INT4/INT8 量化（GPTQ、AWQ）既不用 per-tensor 也不用整条 per-channel，而是**分组（group-wise）**：沿输入特征维把每行切成固定大小 `group_size` 的小块，每块一套 scale。因为一条输出通道沿 K 方向的动态范围也会剧烈波动，per-channel 太粗；分组在「精度」与「scale 存储/开销」间取平衡（`group_size=128` 是常见默认）。本题实现分组对称 fake-quant。

## 规则

实现 `solve(W, group_size, bits=8)`，返回**反量化后的浮点张量**（与 `W` 同形同 dtype）：

1. 记 $q_{max} = 2^{\text{bits}-1} - 1$（bits=8 → 127，bits=4 → 7）。
2. 沿**最后一维**把每个向量切成连续组：第 0 组元素 `[0, group_size)`、第 1 组 `[group_size, 2·group_size)`…… **最后一组允许短于 group_size**（不丢弃、单独统计）。
3. 每组 $g$：$\text{scale} = \max_{i\in g}|W_i| \,/\, q_{max}$；对该组 $q = \operatorname{clamp}(\operatorname{round}(W_i/\text{scale}),\,-q_{max},\,q_{max})$，写回 $q\cdot\text{scale}$。
4. **退化组**（全零）：absmax=0 → 该组原样返回 0（避免除零）。

> 对 `[out, in]` 权重即「每一行独立分组」；`out` 之外还可能有前导维（把除最后一维外的所有位置各自成一行分组）。

## 示例

`bits=8, group_size=4`，$q_{max}=127$：

```
W = [[1.0, -2.0, 0.5, 0.25,  4.0, -1.0]]
组0 absmax=2 → scale=2/127；组1 [4,-1] absmax=4 → scale=4/127
输出 = [[1.00787, -2.0, 0.50394, 0.25197,  4.0, -1.00787]]
```

`bits=4, group_size=4`，$q_{max}=7$：

```
W = [[1.0, 3.0, -7.0, 0.5,  2.0, 2.0, 2.0, 2.0]]
组0 absmax=7 → scale=1 → [1, 3, -7, round(0.5)=0(银行家)] → [1,3,-7,0]
组1 absmax=2 → scale=2/7 → 每个 2→round(7)=7→7·(2/7)=2
输出 = [[1.0, 3.0, -7.0, 0.0,  2.0, 2.0, 2.0, 2.0]]
```

## 约束

- `1 ≤ group_size`（可大于最后维长度，此时整行是一组）；`bits ∈ {2,…,16}`
- `round` 用银行家舍入（同 `torch.round`）；不修改输入、输出 dtype 与输入一致
- 最后一组的短尾必须**按自己长度**独立估 scale，别按满组补零（补零会错误抬高 absmax）
