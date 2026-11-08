# 851 解答笔记：Captum NoiseTunnel 加噪平滑账

## 1. 官方回源

### 三个源码文件

从 `pytorch/captum` 的 `main` 分支拉取三份源码：`noise_tunnel.py`（504 行，含 `NoiseTunnel.attribute` 与全部嵌套函数）、`common.py`（927 行，含 `_expand_and_update_baselines` 与 `_validate_input`）、`attr_common.py`（386 行，含 `_validate_noise_tunnel_type`）。曾遇到 TLS / upstream 失败导致文件被覆盖成一行错误文本，用诊断请求重新拉取后修复。

### 逐字执行环境

判题环境没有 `captum` 包，因此新建 `official_nt.py`：注入最小的假 captum 模块，补齐注解名与依赖，按 `common.py -> attr_common.py -> noise_tunnel.py` 的顺序逐字 `exec` 三份官方源码；再提供 `_FakeMethod` 作为底层归因方法，并模拟官方 `log_usage` 的 `__wrapped__`。这样探针跑的是真正的官方实现，而不是重写版。

## 2. 十段官方机制探针

### 探针结论

`probe851.py` 十段全部跑通，钉死以下机制：支持的三种 `nt_type` 与 `assert` 校验；`repeat_interleave` 的连续重复布局与逐元素独立高斯采样；`stdevs` 的 `float` / `int` / `tuple` 分支；`view(bsz, k, d)` 后 `sum(dim=1)` 与逐元素平方和的分离累加；三种聚合口径；`min` 与整除分批；`delta` 的 `cat` 拼接；`stdevs=0` 的确定性；baseline 的 `replace=True` 抽样与默认扩展条件；`is_delta_supported` / `is_gradient_method` 的继承。

```text
SUPPORTED_NOISE_TUNNEL_TYPES = ['smoothgrad', 'smoothgrad_sq', 'vargrad']
```

## 3. 三层对拍

### official 层

用 `official_nt.py` 逐字执行官方 `NoiseTunnel.attribute`；用 spy 替换 `_FakeMethod.attribute` 捕获每个 partition 的 perturbed（即底层归因方法的真实入参）；baseline 分支直接调用官方命名空间里的 `_expand_and_update_baselines`。

### reference 层

即本题参考解 `solution.py`，五个 mode 全部走 `solve(mode, **kw)`。

### oracle 层

在 `verify851.py` 里独立重写纯 Python 公式，不 import 参考解的任何函数：分批算式、按 partition 分段累加、三种聚合、加噪布局、baseline 两条分支、delta 拼接全部独立实现。

### 对拍结论

五个 mode 共 4843 个检查项，official-vs-reference 用 `1e-4` 容差（官方 float32 与参考解 float64 的精度差），oracle-vs-reference 用 `1e-9`，域约束与官方 quirk 对照另行断言，最终 ALL GREEN。

## 4. 官方 quirk 清单

### stdevs tuple 真 bug

`stdevs_` 只在 `float` 分支赋值，`tuple` 分支的 `zip(inputs, stdevs_)` 在长度匹配时抛 `UnboundLocalError`；源码的 pyre-fixme[61] 已承认。参考解统一显式抛 `ValueError` 并说明这是官方 bug。

### per-partition 加噪布局

主循环 `for _ in range(partition)` 每次独立调用 `add_noise_to_inputs(k)`，`remaining` 分支再单独调一次。因此整体 perturbed 是 `concat_p(repeat_interleave(inputs, k_p))`，不是一次性 `repeat_interleave(inputs, nt_samples)`。三层对拍首轮曾在 `stdevs=0` 用例上暴露这个差异。

### delta cat 与文档不符

文档声称 delta 是「arithmetic mean across all nt_samples」，实现是 `torch.cat(delta_partial_list, dim=0)`，长度 `bsz * nt_samples`。

### 默认 baseline 扩展条件

只在 `baseline.shape[0] == input.shape[0]` 且 `> 1` 时 `repeat_interleave`；draw 分支的校验只要求特征维一致。

```text
delta = torch.cat(delta_partial_list, dim=0)
```

## 5. 五个 mode 的实现要点

### noise_layout

`_validate_stdevs` 先排除 `bool`，再放行 `float`，`int` 与 `tuple` 分别抛带关键字的 `ValueError`；`stdev_expanded` 是标量广播；`repeat_alt` 与 `repeat_alt_source` 作为错误候选一并返回，供双向断言使用。

### batch_schedule

`eff` 用 `min`；`partition` 用 `//`；`remaining` 单独追加；`total` 是 `sum(batch_sizes)`，恒等于 `nt_samples`。

### aggregate

`_accumulate` 按 `batch_sizes` 分段，每段 `[bsz*k][d]`，第 `bi` 个样本占 `part[bi*k:(bi+1)*k]`；`sum` 与 `sum_sq` 分开跨 partition 累加；最后统一除以 `nt_samples`；`vargrad` 是 `E[X^2] - E[X]^2`。

```text
expected_attributions    = sum_attributions    * 1 / nt_samples
expected_attributions_sq = sum_attributions_sq * 1 / nt_samples
```

### baseline_expand

draw 分支要求 `baseline_indices` 长度为 `n_samples * bsz`，逐项校验范围，按索引取 baseline；默认分支按 `num_ref == bsz and bsz > 1` 判定 `repeat_interleave` 或 `unchanged`。

### noise_tunnel

按 `batch_sizes` 逐 partition 构造 perturbed 并累加噪声偏移；`_accumulate` 复用同一分批；`return_convergence_delta` 时校验 `delta_parts` 数量与每 part 长度，再 `extend` 成 `cat` 语义。

```text
vargrad = E[X^2] - E[X]^2
```

## 6. 域约束与错误消息

所有非法输入统一抛 `ValueError`，且消息点名被违反的约束：`stdevs` 的 `float` / `official bug` / `must equal`、未知 `nt_type` 的 `must be either`、`nt_samples` 与 `bs` 的 `positive integer`、行数 `divisible`、indices 的 `must be provided` / `6 elements` / `out of range`、`delta_parts` 的 `must be provided` / `one entry per batch partition`、未知 mode 的 `unknown mode`。判题用例不只断言「抛异常」，还断言消息内容，避免删掉检查后让下游形状校验代为报错。

## 7. 判题用例设计

### 独立 oracle

每个 check 自带按官方算式重写的纯 Python oracle，对每个返回字段逐一比对；数据用确定性 LCG 生成，不依赖任何随机源。

### 双向断言

除等于 oracle 外，还断言结果必须不等于「看起来也合理」的错误候选：`repeat_interleave` vs `repeat` 交替、`sum_sq` vs `(sum)^2`、`smoothgrad_sq` vs `E[X]^2`、`vargrad` vs Bessel 校正样本方差、`delta` vs 广播平均、per-partition perturbed vs 一次性整体布局。注意 `bsz=1` 或 `k=1` 时两种布局天然相同，断言必须挑 `bsz>1` 且 `k>1` 的数据。

### 域约束用例

非法输入的形状必须合法，确保删掉目标检查后不会被下游形状校验顶替报错；同时断言错误消息里的关键字。

## 8. mutation 覆盖

`mut851.py` 共 76 项变异，覆盖两种 repeat 布局、`stdevs` 六条分支、`nt_type` 校验、`_batch_schedule` 九处算式、`_accumulate` 十七处累加与别名、五个 mode 的返回字段与编排、以及 `solve` 分派。首轮 3 项因锚点文本重复报 ANCHOR-ERROR，改用带上下文的唯一锚点后重跑：**76 CAUGHT / 0 SURVIVED / 0 ANCHOR-ERROR**。

## 9. 常见踩坑

- 判题用例必须写 `check_*(solve)`，不能 `from solution import solve`；mutation runner 也要同步用 `getattr(T, name)(S.solve)`。
- 官方是 float32、参考解是 float64，official-vs-reference 容差要放到 `1e-4`。
- 域约束只断言异常类型不够，必须断言消息关键字，否则删检查能存活。
- 双向断言本身要挑数据：`bsz=1` / `k=1` / 零方差数据都会让错误候选与正确结果相等。
- `stdevs=0` 的端到端用例必须按 per-partition 布局构造 attributions，否则 `smoothgrad` 会算成样本混合均值。

## 10. 自检清单

- 三层对拍 4843 项 ALL GREEN。
- 判题用例 5/5 通过参考解。
- mutation 76/76 全 CAUGHT。
- 双语校验零新增 error / warning。
- selftest run 与 submit 双模式全 ACCEPTED。
