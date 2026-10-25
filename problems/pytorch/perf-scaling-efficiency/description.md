# 强扩展与弱扩展：加速比、效率与饱和点

## 题目描述

「8 张卡是不是就快 8 倍？」Scaling 曲线要回答两件事：**加速比（speedup）**——比基线快多少；
**效率（efficiency）**——加速比除以卡数倍数，理想是 1。两种扩展方式的口径不同：
**强扩展（strong scaling）** 保持总工作量不变（卡越多每卡干得越少），**弱扩展（weak scaling）** 保持
每卡工作量不变（卡越多总工作量越大）。效率 $>1$ 是**超线性**（常因分片后每卡的通信/显存压力骤降），
吞吐不再上升的那一档是**饱和点**——再加卡只是多烧钱。

`distributed-pipeline-bubble` / `distributed-zero-communication` 解释**为什么**会掉效率（气泡、通信），
本题只管**怎么算**这条曲线。

## 规则

`world_sizes` 是升序正整数列表 $w[i]$（各档卡数），`step_times_ms` 是等长的正数列表 $t[i]$
（该档**每步墙钟耗时**，ms），第 0 档是基线。

- **整机吞吐（相对量）**：
  - `mode == "strong"`：每步处理的总样本数固定 $\Rightarrow$ 吞吐 $\propto 1/t_i$
  - `mode == "weak"`：每卡处理的样本数固定 $\Rightarrow$ 每步总样本 $\propto w_i$，吞吐 $\propto w_i/t_i$
  $$ \text{speedup}_i=\frac{\text{throughput}_i}{\text{throughput}_0},\qquad
     \text{efficiency}_i=\frac{\text{speedup}_i}{w_i/w_0} $$
  等价闭式：strong $\text{speedup}=t_0/t_i$、$\text{efficiency}=(t_0/t_i)(w_0/w_i)$；
  weak $\text{speedup}=(w_i t_0)/(w_0 t_i)$、$\text{efficiency}=t_0/t_i$。
  两种模式都恒有 $\text{speedup}_i=\text{efficiency}_i\cdot(w_i/w_0)$，且 $\text{speedup}_0=\text{efficiency}_0=1$。
- `superlinear`：$\text{efficiency}_i>1.0$ 的下标（**严格**大于；$\text{efficiency}=1$ 不算），升序 `int64` 张量。
- `saturation_index`：最小的 $i\ge1$ 使 $\text{throughput}_i\le\text{throughput}_{i-1}$（加卡不再提高整机吞吐，
  含持平）；整条曲线严格上升则返回 `-1`。

## 示例

```
输入：world_sizes=[1,2,4,8,16], step_times_ms=[100,45,25,15,9], mode="strong"
  speedup    = [1, 100/45, 100/25, 100/15, 100/9] = [1.0, 2.2222…, 4.0, 6.6667…, 11.1111…]
  efficiency = speedup / w = [1.0, 1.1111…, 1.0, 0.8333…, 0.6944…]
  superlinear=[1]（2 卡那档效率 >1；4 卡恰为 1.0 不算）；吞吐单调上升 -> saturation=-1
输出：(tensor([1.0, 2.2222…, 4.0, 6.6667…, 11.1111…]),
        tensor([1.0, 1.1111…, 1.0, 0.8333…, 0.6944…]), tensor([1]), -1)

输入：world_sizes=[1,2,4,8], step_times_ms=[100,105,115,240], mode="weak"
  efficiency = t0/t = [1.0, 0.9524…, 0.8696…, 0.4167…]
  speedup    = efficiency * w = [1.0, 1.9048…, 3.4783…, 3.3333…]
  8 卡吞吐 8/240 低于 4 卡的 4/115 -> saturation=3；无超线性
输出：(tensor([1.0, 1.9048…, 3.4783…, 3.3333…]),
        tensor([1.0, 0.9524…, 0.8696…, 0.4167…]), tensor([]), 3)
```

## 约束

- $1\le$ 列表长度 $\le 1024$；两列表等长；`world_sizes` 严格升序正整数；`step_times_ms` 为正数
- `mode` 取 `"strong"` 或 `"weak"`
- 返回 `(speedup, efficiency, superlinear, saturation_index)`：前两项 `float64` 张量
  （`atol=1e-5, rtol=1e-5`），第三项 `int64` 张量（`torch.equal`），第四项 `int`
- 只按吞吐比口径计算，不建模带宽/拓扑；饱和点含「持平」，超线性不含「恰好等于 1」
