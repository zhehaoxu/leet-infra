# 笔记：LLM-Pruner 泰勒组粒度头重要性

## 上游出处

论文 *LLM-Pruner: On the Structural Pruning of Large Language Models*（2305.11627, NeurIPS 2023）。仓库 `horseee/LLM-Pruner`（2026-10-31 抓取，`main` 分支，184 个 py 文件）。逐行对齐文件：

- `LLMPruner/pruner/hf_llama_pruner.py::TaylorImportance`（L215–344）——一阶泰勒显著性的组打分，含 attention 头分支；
- `LLMPruner/pruner/hf_llama_pruner.py::_reduce`（L221–238）——group_reduction 枚举（sum/mean/max/prod/first/second）；
- `LLMPruner/pruner/llama_pruner.py::AttentionPrunner`（L36–63）——头耦合的物理移除（wq/wk/wv/wo 同时缩，`dim/head_dim` 同步）；
- `LLMPruner/torch_pruning/pruner/algorithms/metapruner.py`——组框架（组内 idx 粒度与 min-size 对齐）；
- `hf_prune.py`（L71–130）——`--taylor` 参数族：`vectorize`（默认 `param_first`）、`param_second`、`param_mix`。

## 逐行口径

1. **显著性（hf_llama_pruner.py L256–266）**：`salience = weight * grad` 逐元素。`param_second` 是 `w·acc_grad·w`、`param_mix` 是 `salience - 0.5·w·acc_grad·w`——需要二阶梯度缓冲（`acc_grad`），纯前向判题环境不可得，**本题只复现 `param_first` 与 `vectorize`**。
2. **attention 分支（L309–327）**：o_proj 按 out 方向（`sum(1)`）、q/k/v 按 in 方向（`sum(0)`）归约；`vectorize` 是 `sum(...).abs()`，`'param' in taylor` 是 `.abs().sum(...)`。
   **上游 bug（L323）**：`elif 'param' in self.taylor == 'param':` 是链式比较 `('param' in self.taylor) and (self.taylor == 'param')`——对 `'param_first'`/'param_second'/'param_mix' 全为 False → `raise NotImplementedError`。即 attention 分支的 param 路径**实际不可达**，只有 `vectorize` 能走通（q/k/v/linear 分支的 L277–278 `elif 'param' in self.taylor:` 无此问题）。notes 如实记录；本题的 `param_first` attention 语义取 L277–278 的同款 `.abs().sum()` 口径在 attention 布局上的自然推广。
3. **组内对齐（L332–337）**：多依赖组的 `local_norm` 长度不一（o 是 d、q/k/v 是 head_dim）时，`len % min_size == 0 → view(len//min, min).sum(0)` 对齐。o 的 d 长向量按头分块求和 = 本题的 `view(n_heads, head_dim).sum(dim=1)`——这正是「每头标量」的来源：**头是 attention 组的物理粒度**（AttentionPrunner 整头移除），块内求和是把 head_dim 个子通道归并到头。
4. **跨源归并（L340–341）**：stack 后 `_reduce`，上游默认 `group_reduction="sum"`（hf_prune.py L89 显式传 `args.grouping_strategy`，默认值 `sum`）。mean 只差常数 4，选择不变。
5. **选择**：MetaPruner 组内按 importance 升序淘汰（`prune_local` 的 `torch.argsort(imp)` 语义）；本题规定升序 stable、并列取小索引，且 `num_prune` 显式参数化（上游由 `ch_sparsity` 换算，判题直接给头数更确定）。

## 建模决策（与上游框架的差异）

- **组框架解耦**：上游 importance 返回 head_dim 长向量、由 torch_pruning 依赖图决定组内 idx 粒度；本题直接定义**每头一个标量**（块内求和后的头聚合），数学上等价于「头粒度组 + 块内 sum 对齐」，避开依赖图追踪。已在 description 与 notes 双语声明。
- **MHA 而非 GQA**：上游 LLaMA-1/2 无 GQA，q/k/v 同布局；本题要求 `d_out == d_in` 并校验。
- **无 `param_second`/`param_mix`**：需要 `acc_grad` 二阶缓冲，判题环境不可得。
- **float64 内部计算**；上游 fp32。

## 测试设计与变异

- 手算用例（4×4 全正）：o 行块 `[36, 100]`、q/k/v 列块 `[60, 76]`、imp `[216, 328]`；
- vectorize 抵消用例：`[1,-3,1,-3]` 列块 param_first 6 vs vectorize 0；
- **选择反转用例**（head_dim=1）：`[3,-3]` vs `[2,0]`，param_first 剪头 1、vectorize 剪头 0——abs 时机是本题核心考点；
- tie 与极端（num_prune=0/全剪）、mean 与 sum 的常数 4 倍关系、随机形状 numpy 独立 oracle 对拍（双模式，rtol 1e-9）；
- 校验 12 条：形状/dtype 一致性、head_dim 整除与非正、mode/reduction 枚举、num_prune 越界、out≠in、一维权重。

变异 8 个：7 抓获（abs-after-sum / sum-after-abs-vec / o-cols-instead-rows / no-salience-mult / grad-only / view-sum-to-mean / prune-largest）；`unstable-tiebreak`（stable=True → argsort 默认）在 torch CPU 后端对并列值同样稳定，属**测试不可区分的等价变异**，已记录——tie-break 语义仍以 stable=True 为准（并行/JIT 后端下 argsort 不保证稳定）。

## 判重声明

与 470 `sparsity-head-importance` 的核心差异：470 是 `||W_h||_F·||X_h||_F`（无梯度、无依赖组）；本题 `w·g` 泰勒乘积 + o/q/k/v 四源耦合 + 双口径。与 582 OWL（层间预算）完全不同轴。
