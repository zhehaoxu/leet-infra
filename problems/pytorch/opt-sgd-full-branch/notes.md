# notes（opt-sgd-full-branch）

## 回源

- 源码：本地判题环境 torch 2.14.1，`torch/optim/sgd.py`（与上游 main 一致）。
  - `SGD.__init__`（28-82）：`lr < 0`（45-46）、`momentum < 0`（47-48）、
    `weight_decay < 0`（49-50）、`nesterov and (momentum <= 0 or dampening != 0)`（63-64）。
  - `SGD.step`（106-148）：按 group 取 `_init_group` 的 `(params, grads,
    momentum_buffer_list)`，算完再写回 `state[p]["momentum_buffer"]`。
  - `_single_tensor_sgd`（322-379）：344 maximize 取负、346-355 weight_decay、
    357-369 动量/nestersov、371-379 参数更新。
  - `_multi_tensor_sgd`（382-460）逐行手核过：同样先 `_foreach_neg`、同样
    「buf 为 None 就 clone、否则 `mul_(momentum).add_(g, 1-dampening)`」，语义一致。
- 对拍：本题 16 条用例里 `check_matches_torch_optim` 直接拿 `torch.optim.SGD` 当
  oracle（9 组配置 × 3 次 × 7 步，其中两组强制 `grad=None`），float64 下逐步
  1e-12 对齐。

## 官方实现里的坑

1. **首个动量步忽略 dampening**：`buf = grad.detach().clone()` 不走
   `alpha=1-dampening` 那条路。所以 `momentum=0.9, dampening=0.5` 的第一步方向就是
   梯度本身，手写成 `buf=(1-τ)g` 立刻错。
2. **全零缓冲 ≠ 没有缓冲**：官方的判据是 state dict 里有没有 `momentum_buffer`
   这个键，不是缓冲是否为零。本题把它显式成 `buffers=None`（无历史）vs 传入全零
   张量（有历史），两者首步差一个 `(1-dampening)` 因子。
3. **maximize 的取负在 weight_decay 之前**：`grad = grads[i] if not maximize else
   -grads[i]`，之后才 `grad.add(param, alpha=weight_decay)`。有效梯度是
   `-g + λp`，**不是** `-(g + λp)`。`_multi_tensor_sgd` 也是先 `_foreach_neg`。
4. **nesterov 用更新后的 buf，且加的是 g 不是 buf**：`buf.mul_().add_()` 之后
   `grad = grad.add(buf, alpha=momentum)`，`grad` 此刻已含 weight_decay。
5. **momentum == 0 时 dampening 完全不参与**，也不报错——构造期只校验
   `nesterov + dampening` 的组合。
6. **dampening 本身不校验范围**：`5.0`、`-0.1` 都合法（只要 `nesterov=False`）。
   `dampening=1` 时后续步的缓冲只按 `momentum` 衰减，梯度项系数为 0。
7. **nesterov 的门是 `momentum <= 0`**：`momentum=0` 也报；但 `momentum<0` 更早
   就撞上 `Invalid momentum value`，所以第 4 条门实际只对 `momentum==0` 与
   `dampening != 0` 生效。
8. **校验顺序固定**：`lr` → `momentum` → `weight_decay` → `nesterov`。同时给多个
   非法值只看第一个。
9. **`lr == 0` 合法**：参数不动，但缓冲照常推进（没有「lr 为 0 就短路」）。
10. **`p.grad is None` 的参数被 `_init_group` 整个过滤掉**，state 也不建。本题把
    `grads[t] is None` 映射成「整步跳过」——于是**跳过步之后的第一个真梯度会重新
    走一次首建分支**，dampening 又被忽略一次。这是本题最刁的一条：`[None, g]` 与
    `[zeros, g]` 结果不同。
11. **weight_decay 用的是本步更新前的 param 值**（`grad.add(p, ...)` 里的 `p` 还
    没动）。
12. **官方是原地更新**（`param.add_(grad, alpha=-lr)`、`buf.mul_().add_()`）；本题
    不原地改入参，改为 rebind / 新建张量。

## 本题的规范化

- **「按参数过滤」映射成 `grads` 里的 `None`**：官方是单参数视角下 `p.grad is None`
  就跳过，本题把「这一步没有梯度」显式成序列里的 `None`，语义等价。
- **隐藏 state 显式化**：官方把缓冲藏在 `optimizer.state` 里，本题把它做成
  `buffers` 入参 + 返回值；约定 `buffers=None` ⇔ 没有 `momentum_buffer` 键，传入
  张量 ⇔ 已有该键（即使全零）。
- **`dirs` 是新加的观测面**：官方没有对外暴露「这一步到底用了哪个方向」。为了能
  不靠浮点漂移就区分 nesterov / 非 nesterov、首建 / 非首建，本题把 `step` 也记进
  账本，并 clone 后返回（就地改它不影响 `buffers`）。
- **只收 Python 标量 `lr`**：官方允许 1 元素 Tensor（含 requires_grad 的
  differentiable 路径），本题按 `_to_scalar` 之后的语义出题。
- **不建 per-group**：官方 `param_groups` 允许多组超参，本题单组。

## 变异

26 条变异：**22 条实质变异全部 CAUGHT**，4 条判为等价，均有实证（1500 组随机
配置逐字节比对，0 差异）：

- `m-no-param-clone`（`p = params.clone()` → `p = params`）：循环里 `p` 只被
  `p = p - lr*step` **重新绑定**，从不原地改；`path` 又逐段 clone，所以入参不会被
  写到。
- `m-mom-gate-positive`（`momentum != 0` → `momentum > 0`）：`momentum < 0` 在
  `_check` 里就抛 `ValueError`，循环内的负 `momentum` 不可达。
- `m-wd-gate-positive`（`weight_decay != 0` → `> 0`）：同上，负 `weight_decay` 由
  校验拦掉。
- `m-no-grad-clone`（`g = g_in.clone()` → `g = g_in`）：`-g`、`g + wd*p`、
  `(1-dampening)*g` 全是分配新张量，`g` 只被读；返回的 `dirs` 也 clone。

首轮 SURVIVED、补断言后才抓住的：

- `m-dirs-no-clone`（`dirs.append(step.clone())` → `dirs.append(step)`）：`buf` 每步
  都是**重新赋值**（`buf = momentum*buf + ...`）而非原地累乘，所以 `dirs[i]` 拿到的
  旧对象与最终 `buffers` 天然不同；只有「单步 + momentum≠0」时 `step` 就是 `buf`
  本体，两者才共享存储。补了「就地改 `dirs[0]` 后 `buffers` 必须还是原值」的断言
  （`check_no_inplace`）才抓住。

## 查重

- 与 `sgd-momentum-step`（order 74）划界：74 明确 `dampening=0`、`nesterov=False`、
  单步，且 `buffers` 必给（一律按「已有状态」处理），不考 `maximize` 与参数校验；
  本题补全分支矩阵、多步 carry-over、`None` 跳过、首建口径与四道校验门。
- 与 `adam-step`（75）、`opt-adamw`（182）、`opt-adamp`（189）划界：自适应分母、
  偏差修正、解耦权重衰减都不在本题，本题只有一手 SGD。
- 与 `opt-clip-accum`（197）划界：那是梯度累积 + 裁剪，不涉及动量缓冲与 nesterov。
- 全库没有第二个「SGD 分支矩阵 + 构造期参数校验」轴的题目。
