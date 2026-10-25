# 训练 step 时间分解与 Amdahl 上限

## 题目描述

profiler 给出的一个训练 step 通常能拆成四段串行时间：**取数等待（dataloader）/ 前向 / 反向 / 优化器更新**。
拿到这张表要回答两个问题：①「把其中一段优化 k 倍，整机 step 能快多少？」——这是 Amdahl 定律，上限由**没被优化
的那部分占比**卡死；②「取数那段到底能不能靠 `num_workers` + `prefetch` 藏进计算里？」——这是数据管线账。
本题把这两问变成一道纯算术题。`transformer-mfu` 算的是 step 的**算力利用率**、`perf-kernel-launch-timeline` 算
的是 GPU 侧**发射空转**，本题算的是**相位归因 + 提速上限 + 取数掩盖**。

## 规则

`phase_ms` 是 4 个非负数，按序为 `[dataloader_wait, forward, backward, optimizer]`（单位 ms，串行相位口径）。
记 `total` $=\sum_i \texttt{phase\_ms}[i]$，$p=$ `phase_ms[target_phase]`，$k=$ `speedup`。

- **占比**：$\text{fracs}[i]=\texttt{phase\_ms}[i]/\text{total}$（4 元 `float64` 张量，和为 1）。
- **Amdahl**：只有被点名的相位被压缩，其余原样保留：
  $$ \text{new\_step}=(\text{total}-p)+\frac{p}{k},\qquad
     \text{amdahl}=\frac{\text{total}}{\text{new\_step}},\qquad
     \text{ceiling}=\lim_{k\to\infty}\text{amdahl}=\frac{\text{total}}{\text{total}-p} $$
  `ceiling` 若 `total == p`（其余相位全为 0）记 `inf`；`new_step == 0` 时 `amdahl` 记 `inf`。
- **取数掩盖**（与 `total` 是两套视角，不要混用）：$W=$ `num_workers`，供给间隔
  $$ \text{supply}=\frac{\texttt{worker\_ms}}{W},\qquad \text{compute}=\texttt{phase\_ms}[1]+\texttt{phase\_ms}[2]+\texttt{phase\_ms}[3] $$
  `prefetch` $=0$ 表示同步取数、完全没有重叠：$\text{wait}=\text{supply}$；
  `prefetch` $\ge 1$ 时队列至多提前攒着，稳态只需补「供不上」的那部分：
  $$ \text{wait}=\max\!\big(0,\ \text{supply}-\text{compute}\big),\qquad
     \text{effective}=\text{compute}+\text{wait} $$
  `data_hidden` $=(\text{wait}=0)$（**恰好** supply $=$ compute 算掩盖成功）。
  `effective` 是**稳态每步摊销**，首个 batch 的暖机不计。

## 示例

```
输入：phase_ms=[20,100,200,40], target_phase=0, speedup=4, worker_ms=80, num_workers=4, prefetch=2
  total=360; fracs=[20/360, 100/360, 200/360, 40/360]
  new_step = (360-20) + 20/4 = 345      amdahl = 360/345 = 1.0435
  ceiling  = 360/340 = 1.0588            ← 把 dataloader 优化到 0 也只有 5.9% 收益
  supply = 80/4 = 20; compute = 340; 20 <= 340 → wait=0 → effective=340, hidden=True
输出：(tensor([0.0555…, 0.2777…, 0.5555…, 0.1111…]), 360.0, 345.0, 1.0435…, 1.0588…, 340.0, True)

输入：phase_ms=[150,60,120,20], target_phase=0, speedup=8, worker_ms=90, num_workers=3, prefetch=0
  total=350; new_step = 200 + 150/8 = 218.75; amdahl = 350/218.75 = 1.6; ceiling = 350/200 = 1.75
  supply = 90/3 = 30; compute = 200; prefetch=0 → wait=supply=30 → effective=230, hidden=False
输出：(tensor([0.4285…, 0.1714…, 0.3428…, 0.0571…]), 350.0, 218.75, 1.6, 1.75, 230.0, False)
```

## 约束

- `phase_ms` 恰含 4 个非负数且 `total > 0`；`target_phase` $\in\{0,1,2,3\}$；`speedup` $>0$
- `worker_ms` $\ge 0$；`num_workers` $\ge 1$；`prefetch` $\ge 0$（整数）
- 返回 `(fracs, total_ms, new_step_ms, amdahl_speedup, ceiling_speedup, effective_step_ms, data_hidden)`：
  第 1 项是 `float64` 张量，第 2-6 项是 float（`atol=1e-5, rtol=1e-5`），第 7 项是 bool
- 相位按串行口径相加，不涉及多流重叠；`fracs` 用 `phase_ms[0]`（实测等待）而非 worker 模型
