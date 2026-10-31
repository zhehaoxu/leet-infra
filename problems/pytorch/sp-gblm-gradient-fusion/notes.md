# 笔记：GBLM 梯度融合显著性逐行剪枝

## 上游出处

论文 *GBLM-Pruner: Gradient-based Language Model Pruner*（2311.04902, ICLR 2024）。仓库原列 `IST-DASLab/GBLM-Pruner`，已迁移至 **`VILA-Lab/GBLM-Pruner`**（2026-10-31 抓取，`main` 分支）。逐行对齐文件：

- `lib/prune.py::prune_gblm`（L216–311）——核心融合公式与掩码生成；
- `lib/prune.py::prune_wanda`（L314–401）——对照基线，逐行结构与 `prune_gblm` 相同，仅缺梯度项；
- `gradient_computation.py::gradient_computation`（L119–157）——梯度两段式聚合；
- `lib/prune.py::return_given_alpha`（L151–157）——`use_variant` 二分分支（本题**未复现**，见下）。

## 逐行口径

1. **梯度聚合（gradient_computation.py L155–156、L207–209）**：逐样本把 `g·scale` 的绝对值累进两个缓冲：
   - `gradients_l1 += |g·scale|`（fp16 累积）；
   - `gradients_l2 += (g·scale)²`（fp32 累积），落盘前统一 `sqrt`。
   `prune_gblm` 实际加载的是 **L2 落盘文件**（`main.py --prune_method gblm` 默认传 `gradients_aggregrate_norm_l2_*`），所以 `G = sqrt(Σ_s (g_s·scale)²)`。L1 文件存在但主流程不用。
2. **融合（prune.py L259–267）**：
   ```python
   W_metric = |W| * torch.sqrt(scaler_row.reshape((1,-1)))     # L259，Wanda 项
   W_metric_grad = |W| * |G|                                    # L262
   W_metric = W_metric + W_metric_grad                          # L263，非 inv
   # inv 分支 L267：W_metric = W_metric * (1/(|G| + 1e-8))
   ```
   注意梯度项**不带** `√s_row` 因子、直接乘 `|W|`——这是「列激活 RMS + 逐元素梯度」两个证据的加法融合，不是 `|W|·(√s + G)` 的括号合并（数学等价但括号写法易让人误以为 G 也要乘 `√s`）。`small_value = 1e-8` 只出现在 inv 分支。
3. **scaler_row（layerwrapper.py，与 SparseGPT/Wanda 同族）**：`scaler_row += x²` 逐 batch 累加（sum 口径，未除样本数）。本题沿用 581 题的 **mean 口径**并已在 description 声明——`sqrt(sum) = sqrt(mean·n_tokens)`，`n_tokens` 是行内常数因子，作用于整列，**掩码不变**，仅 `metric` 数值缩放；对拍按 mean 口径。
4. **掩码（prune.py L277–301）**：`torch.sort(W_metric, dim=-1, stable=True)` 后取 `sort_res[1][:, :int(cols·sparsity)]` —— **逐行等量截断**（Wanda 同款），不是全局阈值。stable=True 保证并列值取小索引。
5. **2:4 分支（L272–275）**：与 Wanda 完全同款（每 m 块 topk n 最小），只是作用域换成融合 metric。
6. **`use_variant` 二分分支（L279–297）**：按行累积和做 alpha 二分逼近目标稀疏度（Wanda 变体）。涉及 while 收敛与浮点比较，确定性差，**本题不复现**；`prune_gradient`（L182–213，纯梯度无激活项）与 `prune_magnitude`（L159–180）同样只作对照未复现。

## 与 581 Wanda 题的差异点（判重声明）

| 维度 | 581 Wanda | 本题 GBLM |
| --- | --- | --- |
| 显著性 | `\|W\|·√mean(x²)` | 上项 ⊕ `\|W\|·sqrt(Σ(g·scale)²)` |
| 梯度输入 | 无 | 逐样本梯度 `(S, out, in)` + `scale` |
| 变体 | 无 | `gradient_inv`（`base/(G+1e-8)`） |
| 考点 | 激活 RMS × 逐行截断 | 两证据融合的权重比、L2 vs L1 聚合、scale 敏感性、inv 反转 |

掩码骨架（stable sort 逐行 int 截断、N:M 窗口）与 581 共用，但判题核心是融合公式的四个易错点：聚合方式、scale 缺失、inv 分母 eps、inv 作用基。变异测试 7/7 全部抓获（grad-l1-agg / no-scale / grad-term-missing / inv-no-eps / inv-on-wrong-base / sort-descending / sum-scaler）。

## 有意偏差（与上游不同之处）

- **fp16 → float64**：上游权重与梯度缓冲用 fp16/fp32 混合累积，判题环境统一 float64 对拍，语义等价（融合公式逐元素、无累积次序敏感）。
- **scaler_row mean 口径**：见上第 3 条，掩码不受影响。
- **`use_variant` alpha 二分、`prune_gradient`、`prune_magnitude` 不复现**：非 GBLM 主贡献且含收敛循环。
- **无数据管线**：上游跑 wikitext/c4 前向取激活与反向取梯度；判题直接给 `activations` 与 `grad_samples` 张量，前向/反向不进 solve。
- **scale 显式参数化**：上游命令行默认 100；判题用例显式给定（含 1、10 等小值以暴露权重比效应），语义一致。

## 测试设计

- 手算用例：2×4 双行，覆盖 `√s_row` 有零列、梯度有零列、并列值 stable 序；
- scale 敏感用例：同一输入 scale=1 与 10 掩码不同（证明 scale 不是可省参数）；
- inv 反转用例：非 inv 与 inv 剪不同的列；零梯度位 metric=1e9 巨值不剪（8/20=0.4 反而先剪）；
- 2:4 窗口用例：metric 并列值 `[2,2]` 的 stable tie-break；
- 多样本 L2 聚合 + 随机形状 numpy 独立 oracle 对拍（rtol 1e-9）；
- 校验用例 11 条：三维 weight、列不匹配、dtype 不一致、grad 形状/样本数、sparsity 越界、scale 非正、非布尔 inv、N:M 三约束。
