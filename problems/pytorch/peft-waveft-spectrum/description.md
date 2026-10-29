# PEFT：WaveFT 稀疏频谱增量权重

## 题目描述

**WaveFT**（peft `tuners/waveft.py`）不学权重矩阵本身，而是学**权重的频谱系数**：把 \(\Delta W\) 看成某组变换系数，训练时只更新其中 \(n_\text{frequency}\) 个被随机选中的位置，其余恒为 0。与 LoRA 的"低秩"约束不同，WaveFT 是**坐标稀疏**约束——参数量等于 \(n_\text{frequency}\)，与秩无关，可以远小于 \(r(d_\text{out}+d_\text{in})\)。

本题只覆盖 peft `get_delta_weight` 的 **direct 路径**（`use_idwt=False`）：频谱系数直接按坐标填进一个与 \(W_0\) 同形的稠密矩阵，再乘 `scaling` 得到增量；适配器的前向就是在基座前向上再加一项 \(x\Delta W^\top\)。

## 规则

- 坐标由种子决定：用 `torch.Generator().manual_seed(random_loc_seed)` 配 `torch.randperm(d_out*d_in)[:n_frequency]` 取扁平下标，再按 `row = idx // d_in`、`col = idx % d_in` 落位（行优先展开）。
- \(\Delta W[\text{row},\text{col}] = \text{spectrum}[k]\)，最后整体乘 `scaling`。
- 前向：\(y = xW_0^\top + b_0 + x\Delta W^\top\)；`b0` 为 None 时省略偏置项。
- 内部 float64 计算，返回 float32。

## 示例

\(W_0(3\times4)\)、\(n_\text{frequency}=4\)、种子 777、`scaling=25.0`：

```
输入：
  W0 = [[0.5,-0.2,0.1,0.3],[0.2,0.4,-0.1,0.0],[-0.3,0.1,0.5,0.2]]
  b0 = [0.1,-0.2,0.05]
  x = [[1.0,0.5,-0.3,0.2],[0.1,-0.4,0.6,0.3]]
  spectrum = [0.02,-0.01,0.03,0.015]
输出：
  delta = [[0.0,0.0,0.0,0.75],[0.0,0.0,0.0,-0.25],[0.375,0.0,0.0,0.5]]
  y     = [[0.68,0.18,0.165],[0.605,-0.475,0.5275]]
```

## 约束

- `1 <= n_frequency <= d_out*d_in`，`spectrum` 一维且长度恰为 `n_frequency`；越界或形状不符抛 `ValueError`。
- `x` 支持任意前置维，最后一维为 \(d_\text{in}\)。
- 退化性质：频谱全零时增量严格为零，前向回到基座；`n_frequency = d_out*d_in` 时每个格子恰好落一个系数（增量是频谱的一个置换）。
- 依赖 `use_idwt=True` 的小波反变换路径不在本题范围内（peft 默认走 IDWT，需要 `waverec2d`）。
