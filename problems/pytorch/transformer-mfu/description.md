# Transformer FLOPs 与 MFU（算力利用率）

## 题目描述

**MFU（Model FLOPs Utilization）** 衡量训练实际用到的算力占硬件峰值的比例：把一个训练 step
「应该做」的理论 FLOPs 除以「实测耗时」得到 achieved，再除以芯片峰值即得利用率，是评估
训练系统效率的核心指标。本题按一个标准 Transformer 训练 step 估算前向/整体 FLOPs 与 MFU。

## 规则

- 实现 `solve(b, s, layers, hidden, ffn, peak_tflops, step_ms)`，返回 `(forward_flops, step_flops, mfu)`。
- **每层参与矩阵乘的参数量**（不含 LayerNorm 权重与词嵌入，本简化模型）：
  $$ P_{\text{layer}} = \underbrace{4h^2}_{\text{attn } q,k,v,o} + \underbrace{2 h f}_{\text{FFN 上/下投影}} $$
- **每 token 每层的前向 FLOPs**（一次乘加计 2 FLOPs）：
  $$ c = 2\,P_{\text{layer}} + \underbrace{4\,s\,h}_{QK^\top \text{ 与 } AV \text{ 的非参数 FLOPs}} $$
  其中 $4sh$ = $QK^\top$ 的 $2sh$ + $A V$ 的 $2sh$（对所有头求和后，每 token 对该层 $s$ 个键值做 $h$ 维运算）。
- **一个 step 的前向 FLOPs**（tokens $= b\cdot s$）：
  $$ \text{forward\_flops} = b\cdot s\cdot \text{layers}\cdot c $$
- **整体 step FLOPs**：反向约为前向的 2 倍，故
  $$ \text{step\_flops} = 3 \times \text{forward\_flops} $$
  （这与常见的「训练 $\approx 6ND$」口径一致：前向 $2N$、反向 $4N$ 每 token）。
- **MFU**：`peak_tflops` 为单卡峰值（TFLOP/s），`step_ms` 为该 step 实测耗时（毫秒）：
  $$ \text{achieved} = \frac{\text{step\_flops}}{\text{step\_ms}/1000}\ \text{FLOP/s},\qquad \text{mfu} = \frac{\text{achieved}/10^{12}}{\text{peak\_tflops}} $$

## 示例

```
输入：b=4, s=1024, layers=6, hidden=512, ffn=2048, peak_tflops=150.0, step_ms=8.0
  P_layer = 4*512² + 2*512*2048 = 3,145,728
  c = 2*3,145,728 + 4*1024*512 = 8,388,608   （每 token 每层）
  forward_flops = 4*1024*6*8,388,608 = 206,158,430,208
  step_flops    = 3*forward = 618,475,290,624
  achieved = 618475290624 / 0.008 = 7.7309e13 FLOP/s = 77.309 TFLOP/s
  mfu = 77.309 / 150.0 = 0.5154
输出：(206158430208, 618475290624, 0.5154...)
```

## 约束

- `b, s, layers, hidden, ffn` 为正整数；`peak_tflops`、`step_ms` 为正浮点
- `forward_flops`、`step_flops` 返回整数；`mfu` 为浮点比例（0~1 量级）
- 本简化模型不计 embedding/LayerNorm/激活函数等非 matmul FLOPs（占比小），聚焦主流估算口径
