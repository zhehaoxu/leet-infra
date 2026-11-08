# 853 解答笔记：Captum GradientShap 期望梯度账

## 1. 官方回源

### 三份源码与依赖复用

从 `pytorch/captum` 的 `main` 分支拉取 `captum/attr/_core/gradient_shap.py`（470 行 / 22110 bytes）、`captum/_utils/gradient.py` 与 `captum/attr/_utils/attribution.py`；`noise_tunnel.py` / `common.py` / `attr_common.py` 直接复用 order 851 的暂存源码。`attribution.py` 原 URL 多次超时，改用带查询串的 raw 来源拿到 24275 bytes / 536 行，并用网页交叉核对。

### 逐字执行环境

新建 `official853.py`：注入 14 个 fake Captum modules，把 `ModuleOrModuleList` 设成 `TypeVar` 以满足 `Generic[...]`，用 `functools.wraps` 复刻 `log_usage` 的 `__wrapped__`，按 `common -> attr_common -> gradient -> attribution -> noise_tunnel -> gradient_shap` 顺序逐字 `exec` 并共享命名空间。`SeqRNG` 替换 `np.random.choice` / `np.random.uniform` / `torch.normal`，记录 `choice_calls` / `uniform_calls` / `normal_calls` 与 `order`。冒烟确认命名空间有 112 个符号，四类官方符号齐全。

## 2. 十二段官方机制探针

### 探针结论

`probe853.py` 十二段全部跑通，钉死：`GradientShap.attribute` 硬编码 `nt_type="smoothgrad"` 与 `draw_baseline_from_distrib=True` 且不透出 `nt_samples_batch_size`；`_format_callable_baseline` 三种形态与 `baselines[0]` 必须是 Tensor 的断言；`_expand_and_update_baselines` 的扁平 `k*bsz` 索引与非 distrib 的 `repeat_interleave(k)`；`_validate_input` 只校验 `shape[1:]`；`_scale_input` 的 `(n,1,1,..)` 广播与 `requires_grad_()`；RNG 顺序 choice -> normal -> uniform 与 numpy / torch 双流；`repeat_interleave(k)+normal` 布局；`stdevs` 的 float / tuple 约束；`multiply_by_inputs` 两分支；smoothgrad 单 partition 均值；delta 口径。

```text
NS keys: 112
choice_calls: [(4, 3, True)]
uniform_calls: [(0.0, 1.0, 3)]
normal_calls: [(3, 3)]
order: [('choice', 3), ('normal', 9), ('uniform', 3)]
```

P7 / P11 首轮曾为 False，诊断显示布局完全正确、数值只差约 `1e-8`，根因是官方 `torch.tensor(stdev)` 默认 float32。

## 3. 三层对拍

### official 层

`official853.py::run_gradient_shap` 执行真正的官方 `GradientShap.attribute`，输入用 float64 张量，RNG 由 `SeqRNG` 显式序列驱动。

### reference 层

即本题参考解 `solution.py`，五个 mode 全部走 `solve(mode, **kw)`，纯 Python float64。

### oracle 层

`verify853.py` 用 `Fraction` 重写精确算术，不 import 参考解的任何函数：基线抽样、加噪、插值、解析梯度、乘子、smoothgrad 均值、delta 全部独立实现。

### 对拍结论

12 个用例覆盖 single / tuple、linear / quadratic、mul 开关、delta 开关、不同 bsz / d / k / num_ref，共 340 个检查项，最终 ALL GREEN。

```text
checks: 340   failures: 0
RESULT: ALL GREEN
```

## 4. 官方 quirk 清单

### 单 partition 与 RNG 双流

`GradientShap.attribute` 调用 `NoiseTunnel.attribute.__wrapped__` 时没有传 `nt_samples_batch_size`，恒 `None` 故单 partition。choice 与 uniform 共享 numpy 全局流，normal 走独立 torch 流；tuple inputs 时 choice / normal 每 tensor 一次、uniform 只有一次。

### float32 量化误差

官方 `add_noise_to_input` 的 `torch.tensor(stdev)` 不带 dtype，默认 float32。探针 P7 / P11 因此出现约 `1e-8` 偏差。解法是驱动临时提升全局默认 dtype 并在 finally 恢复，官方源码本身不改。

```text
_prev_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    ...
finally:
    torch.set_default_dtype(_prev_dtype)
```

### 基线索引布局

`np.random.choice(num_ref, k*bsz, replace=True)` 给的是扁平 `k*bsz` 个索引；`_validate_input` 在 distrib 模式只校验 `shape[1:]`。非 distrib 的 `applicable` 必须同时满足 `num_ref == bsz` 与 `num_ref > 1`。

## 5. 五个 mode 的实现要点

### baseline_draw

distrib 走 `rows[i] for i in indices`，`drawn_shape = [k*bsz, d]`；非 distrib 按 `applicable` 分 `repeat_interleave` / `unchanged` 两支，并返回 `non_distrib` 明细。

### scale_input

`coeff_shape = [n] + [1]*(ndim-1)`，插值用 `alpha*x + (1-alpha)*b`，返回 `requires_grad=True` 标记。

### noise_layout

按 `(i*k+j)*d + t` 的 C order 取噪声，`expected = bsz*k*d*n_in`，`_stdevs_check` 统一校验 float / 等长 float tuple。

### schedule

`effective = k`、`partitions = 1`、`remaining = 0`，三组 RNG 调用账与 `rng_order`，`compare_batch_size` 给出直接 NoiseTunnel 的分片对照。

### gradient_shap

逐 tensor 切 `idx_t` / `noi_t`，按抽样 -> 加噪 -> 插值 -> 梯度 -> 乘子 -> 均值串联；delta 跨 tensor 累加 attr 行和、只减一次 F 差。

## 6. 域约束与错误消息

所有非法输入统一抛 `ValueError` 且消息点名被违反的约束：`n_samples must be positive` / `indices length` / `indices must be ints` / `baseline index out of range` / `baselines rows must match inputs rows` / `baselines feature dim must match inputs feature dim` / `alpha must come from np.random.uniform` / `ndim must be` / `k must be positive` / `n_in must be positive` / `noises length` / `noises must be finite numbers` / `stdevs must be type float` / `number of stdevs` / `each stdev` / `alphas must be in` / `multiply_by_inputs must be bool` / `all input tensors must share bsz and feature dim` / `tuple lengths must match` / `unknown mode`。

```text
noises must be finite numbers
baseline index out of range [0, 4)
```

## 7. 判题用例设计

### 独立 oracle

8 个 `check_*(solve)` 中 `check_gradient_shap` / `check_convergence_delta` 用独立重写的 `_exp_gs`（逐样本显式循环 + 多项式展开），对每个返回字段逐一比对；`_case` 生成确定性序列，无随机源。判题用例不 `import solution`，`check_*` 必须接受 `solve` 参数。

### 双向断言

除等于 oracle 外还断言结果必须不等于「看起来也合理」的错误候选：`repeat()` 布局、alpha 上界 `1.0`、mul 开关下 grads 与 attr 的关系、单 partition 与多 partition、tuple 时第二段 noise / indices 切片复用。

### 域约束用例

非法输入的形状必须合法，确保删掉目标检查后不会被下游形状校验顶替报错；同时断言消息关键字。`num_ref == bsz == 1` 钉死非 distrib 的 `applicable`，`bsz > 1 且 k > 1` 钉死 C order 布局。

## 8. mutation 覆盖

`mut853.py` 125 项变异，覆盖公共工具、baseline_draw、scale_input、noise_layout、schedule、gradient_shap 与 solve 分派。首轮 110 CAUGHT / 14 SURVIVED / 1 ANCHOR-ERROR：根因是 `_case` 的噪声序列 `(i*5+seed)%5` 恒为常数，使多 tensor 切片与 `(i*k+j)` 布局变异不可观测；改成 `(i*3+seed)%7` 后补齐逐元素差异，另补 12 处域约束 / 元数据断言，并修 `gs-delta-flag` 因中间夹注释行导致的锚点失配。重跑全绿。

```text
125 effective mutants: 125 CAUGHT / 0 SURVIVED / 0 ERROR
```

## 9. 常见踩坑

- 判题环境没有 `captum`，官方逐字执行必须注入 14 个 fake modules 并处理 `Generic[TypeVar]` 与 `log_usage.__wrapped__`。
- 官方 `torch.tensor(stdev)` 默认 float32，必须临时提升全局默认 dtype 而不是改官方源码。
- `SeqRNG` 的 choice 与 uniform 共享 numpy 流，normal 独立，顺序不能写反。
- 常数噪声序列会让多 tensor 切片与布局变异退化为真等价假象，判题序列必须逐元素变化。
- 判题用例写 `check_*(solve)`，mutation runner 同步用 `getattr(T, name)(S.solve)`。
- 域约束只断言异常类型不够，必须断言消息关键字。
- 锚点必须唯一，跨行锚点若中间夹注释行会失配。
- 等价变异要保留记录但剔除统计。

## 10. 自检清单

- 三层对拍 340 项 ALL GREEN。
- 判题用例 8/8 通过参考解。
- mutation 有效 125 项全 CAUGHT。
- 双语校验零新增 error / warning。
- selftest run 与 submit 双模式全 ACCEPTED。
