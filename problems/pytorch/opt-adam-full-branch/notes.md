# notes（opt-adam-full-branch）

## 回源

- 源码：本地判题环境 torch 2.14.1，`torch/optim/adam.py`（与上游 main 一致）。
  - `Adam.__init__`（34-120）：`not 0.0 <= lr`（58）、eps（60）、betas[0]（62）、
    betas[1]（64）、weight_decay（66），外加 68-71 的「betas 必须同为 float 或同为
    Tensor」。
  - `_init_group`（139-208）：state **懒建**（`len(state) == 0` 才建）——step 是
    `torch.tensor(0.0)`（168）、`exp_avg` / `exp_avg_sq` 全零、`max_exp_avg_sq` 全零
    （187，只有 amsgrad 才建）。
  - `_single_tensor_adam`（347-545）标量路径：397 maximize 取负、414 `step_t += 1`、
    426/428 weight_decay、456 `exp_avg.lerp_`、471/475 `exp_avg_sq.mul_().addcmul_()`、
    530-533 偏差修正与 `step_size`、535 `bias_correction2_sqrt`、539 amsgrad 的
    `torch.maximum(..., out=)`、542/544 denom、546 `param.addcdiv_`。
  - `_multi_tensor_adam`（553-）只看语义：同样先取负、同样 lerp/乘加、同样
    `bias_correction2_sqrt` 与 eps 位置。
- 对拍：`check_matches_torch_optim` 直接拿 `torch.optim.Adam` 当 oracle（8 组配置 ×
  4 次 × 8 步，其中两组强制 `grad=None`、一组全 None），float64 下逐步 1e-12 对齐；
  实测最大绝对偏差 8.9e-16。
- 另跑了 1200 组随机配置（lr/betas/eps/wd/amsgrad/maximize/state 全随机）做等价变异
  实证，见下。

## 官方实现里的坑

1. **`step` 在使用前自增**（414 行）：state 里初值是 0，所以第一次更新用 `t = 1`；
   第 `k` 次更新用 `t = k`。写成「先算后自增」就差一整步的偏差修正。
2. **没有梯度的参数根本不建 state**（`_init_group` 里 `if p.grad is not None` 才进
   循环体），所以**跳过步不推进 `step`**。于是 `[None, g]` 的第一条真梯度用 `t = 1`
   的正确偏差修正，而不是 `t = 2`——这是本题最容易被忽略的一条。
3. **五道校验用 `not 0.0 <= x`**：`0.0 <= nan` 为假 → **NaN 也被拒**，且文案里打出
   `nan`。另外 `betas` 要求 `< 1.0`，所以 `beta = 1.0` 非法（不少复现只查了 `>= 0`）。
4. **`eps` 加在 `sqrt(v)/sqrt(bc2)` 之后**（544 行 `.add_(eps)`），没有乘
   `sqrt(bc2)`。所以第一步的位移是 `-g/(|g|+1e-8)` 而不是 `-g/|g|`：`eps=1e-8` 时
   位移是 `0.099999998`，差 2e-9；若按「eps 也做偏差修正」的论文修正式，差是 6.3e-10，
   量级差三倍，用 `atol=1e-12` 的硬编码值就能钉死。
5. **amsgrad 的最大值先更新、后进分母**（539 在 542 之前），并且 state 里初始化成
   **全零**（不是 `v` 的拷贝）。因为 `v` 恒非负，首步 `max(0, v) = v`，所以「初始化为
   零」与「初始化为 v」在首步不可区分——但**只有零初始化**才在后续 `v` 变小时留住
   旧的最大值（本题用「大梯度 → 小梯度」的两步实例把它测出来）。
6. **maximize 取负先于 weight_decay**（397 vs 426）：有效梯度 `-g + λp`，不是
   `-(g + λp)`。用 `p=[1,-2]`、`g=[0.5,-1]`、`λ=0.5` 能构造有效梯度恰为 0 的退化例：
   正确实现参数不动；若顺序反了就会动。
7. **Adam 的 `weight_decay` 是 L2 式**：λp 并进梯度后**同时**进 `m` 与 `v`，于是衰减量
   被 `1/√v` 打折、不同参数衰减幅度不等；`opt-adamw`（182）是解耦式
   `p *= (1 - lr·λ)`，与自适应分母无关。torch 2.x 另有 `decoupled_weight_decay=True`
   开关（本机 2.14.1 存在，走 `param.mul_(1 - lr*weight_decay)`），本题只考默认 L2 路径。
8. **两处偏差修正折叠方式不一致**：`step_size = lr/bc1` 先折叠，`denom` 里
   `1/sqrt(bc2)` 也先折叠，最后 `addcdiv_(m, denom, value=-step_size)`。与
   `lr·m̂/(√v̂+eps)` 代数等价，但 `eps` 没有跟着 `sqrt(bc2)` 缩放（见第 4 条）。
9. **`state` 懒建 = 全有或全无**：一旦建了，`step/exp_avg/exp_avg_sq`（+ amsgrad 的
   `max_exp_avg_sq`）四个键齐全；不存在「只建一半」的状态。
10. **`lr = 0` 合法**：`step_size = 0` → 参数不动，但 `m`、`v`、`step` 照常推进。
11. **`betas` 允许 0**：`β₁=0` 时 `m = g`，`β₂=0` 时 `v = g²`，两个偏差修正都等于 1。
12. **`eps = 0` 也合法**：此时纯 `sqrt(v̂)` 归一化；若某步梯度恰为 0，`v = 0` 且
    `denom = 0`，`m/denom` 就是 `0/0 = nan`——官方不做任何防护。所以本题的
    「`maximize + wd` 让有效梯度恰为 0」那条退化用例保留默认 `eps`，位移才是 0 而不是
    nan。
13. **`betas` 长度不足时官方抛 IndexError**（`betas[1]` 越界），不是 ValueError。

## 本题的规范化

- **`betas` 只收两个 float**：官方允许「同为 Tensor」的一对，并用
  `"betas must be either both floats or both Tensors"` 拒绝混合；本题统一按 float 出题，
  长度不是 2 时抛 `ValueError`（官方是 `IndexError`）。
- **`state` 显式化**：官方的 state 是带标量张量 `step` 的 dict，本题把它做成入参
  dict，`step` 规范成 int（= 已经更新过的步数）；`amsgrad=False` 时 `max_exp_avg_sq`
  仍要求出现在 state 里但**不被读取**，返回值里给全零（官方 state 里根本没有这个键）。
- **`grads[t] = None` 映射官方「按参数过滤」**：单参数视角下这与「该步 `p.grad is
  None`」等价。
- **新增 `steps` 与 `deltas` 两个观测面**：官方不暴露「这一步用了哪个 t」和「实际位移是
  多少」。把两者入账后，「跳过步不推进计数」和「eps 的位置」都能直接断言，不必靠浮点
  漂移去猜。
- **只考实数张量、单 group、Python 标量 lr/eps/wd**：`view_as_real` 复数路径、
  `per-group` 超参、Tensor lr（`capturable`/`foreach` 组合约束）都不在范围内。

## 变异

33 条变异：**29 条实质变异全部 CAUGHT**，4 条判为等价，均有实证（1200 组随机配置：
lr ∈ {0, 0.01, 0.1, 0.3}、betas ∈ 4 组、eps ∈ {0, 1e-3, 1e-8}、wd ∈ {0, 0.01, 0.2}、
amsgrad/maximize 随机、30% 带随机 state，0 差异）：

- `m-no-param-clone`（`p = params.clone()` → `p = params`）：`p` 只被 `p = p + delta`
  **重新绑定**，从不原地改，`path` 又逐段 clone。
- `m-delta-no-clone`（`deltas.append(delta.clone())` → `deltas.append(delta)`）：`delta`
  是每步新建的张量，且 `p` 由它**相加得到新张量**，不与任何返回字段共享存储。
- `m-momentum-lerp`（`β₁m+(1-β₁)g` → `m + (1-β₁)(g-m)`）：数学恒等，只是舍入路径不同，
  实测最大偏差 **1.06e-15**，远在 1e-12 契约内。
- `m-var-order`（`(1-β₂)·g·g` → `g·g·(1-β₂)`）：乘法结合序不同，最大偏差 **4.4e-16**。

首轮没有「实质变异存活」；需要说明的是**用例先写错了一次**：
`check_skip_none_does_not_advance_step` 里把 `path[3]` 当成「跳过步之后仍是原参数」，
实际 `path[3]` 是第二个跳过步的快照（等于 `path[2]`，不是 `path[0]`），基线先挂——
修成 `path[3] == path[2]` 后 16/16 通过。再次印证「变异前必须确认 selftest 是绿的」。

## 查重

- 与 `adam-step`（75）划界：75 是单步、`amsgrad=False`、无 weight_decay；本题是多步 +
  amsgrad 滑动最大值 + L2 衰减 + maximize + 五道校验 + state 恢复。
- 与 `opt-adamw`（182）划界：AdamW 是解耦衰减（`p -= lr·λ·p`，与 `1/√v` 无关），本题是
  L2（λp 并进两个矩、被 `1/√v` 打折），两题构成同一族的对照分支；本题把「为什么 L2 在
  Adam 里不是等比衰减」写进 notes。
- 与 `opt-adamp`（189）、`opt-radam`（187）、`opt-adamuon`（267）、`opt-adabelief` 等划界：
  那些是变体公式（投影/校正项/muon 化），本题是 vanilla Adam 的完整分支与账本。
- 与 `opt-sgd-full-branch`（738）划界：同一「全分支」系列的 SGD 版，没有自适应分母与
  两个矩；两题共用「None 步不建 state」的口径，但后果不同（SGD 是缓冲首建延后，Adam 是
  偏差修正的 t 不涨）。
- 与 `opt-clip-accum`（197）划界：那是梯度累积 + 裁剪，不做矩估计。
