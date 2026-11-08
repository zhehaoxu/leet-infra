# Captum NoiseTunnel 加噪平滑账：repeat_interleave 加噪布局、nt_samples_batch_size 的 min 与整除分批、smoothgrad / smoothgrad_sq / vargrad 三种聚合口径、baseline 的 draw 抽样与 repeat_interleave 条件，以及 delta 的 torch.cat 拼接

## 1. 背景

Captum 的 NoiseTunnel 是包在任意归因方法外面的平滑层：它对输入加高斯噪声、重复若干次，再对多次归因结果做聚合。本题把整条链路拆成五个可独立判题的数值账，全部用纯 Python 浮点实现，随机噪声与抽样索引一律显式入参，避免任何随机源。

### 官方源码位置

- `captum/attr/_core/noise_tunnel.py` 的 `NoiseTunnel.attribute` 与四个嵌套函数
- `captum/attr/_utils/common.py` 的 `_validate_noise_tunnel_type`
- `captum/_utils/common.py` 的 `_expand_and_update_baselines`

## 2. 口径约定

### 输入与输出

所有张量用嵌套列表表示：`inputs` 是 `[bsz][d]`，加噪后的矩阵是 `[bsz*k][d]`。返回值是字典，字段名与官方变量对齐。所有非法输入统一抛 `ValueError`，并在消息里点名被违反的约束。

### 随机性显式化

官方内部用 `torch.normal(0, stdev_expanded)` 与 `np.random.choice` 采样。本题把 `noise` 与 `baseline_indices` 作为显式入参：判题用例给定它们，参考解只做确定性算术。全程 Python float（float64）。

## 3. mode 1: noise_layout

### 加噪布局

官方 `add_noise_to_input` 先把输入按 `repeat_interleave` 连续重复，再逐元素加噪声：

```text
repeat_interleave(x, 3) = [x0, x0, x0, x1, x1, x1]
repeat(x, 3)            = [x0, x1, x0, x1, x0, x1]
```

`source_index` 是 `i // k`，错误候选 `repeat_alt_source` 是 `i % bsz`。`expanded_size` 是 `[bsz*k, d]`，`stdev_expanded` 每个元素都等于标量 `stdev`。

```text
input.repeat_interleave(nt_samples_partition, dim=0) + noise
```

### stdevs 类型分支

官方 `add_noise_to_inputs` 只接受 `float`：`int` 会触发 `assert isinstance(stdevs, float)`，而 `tuple` 分支虽然先 `assert len(stdevs) == len(inputs)`，但 `stdevs_` 只在 float 分支赋值，长度匹配时会抛 `UnboundLocalError`（源码里 pyre-fixme[61] 已承认）。本题把 tuple 合法长度场景统一显式抛 `ValueError` 并说明这是官方 bug，而不是复现 `UnboundLocalError`。

```text
stdevs_ = (stdevs,) * len(inputs)
```

## 4. mode 2: batch_schedule

### 分批算式

`eff = nt_samples if bs is None else min(nt_samples, bs)`，`partition` 是整除而不是向上取整，`remaining` 单独再跑一批：

```text
eff = nt_samples if bs is None else min(nt_samples, bs)
partition = nt_samples // eff
remaining = nt_samples - partition * eff
batch_sizes = [eff] * partition + ([remaining] if remaining > 0 else [])
```

`bs > nt_samples` 会被 `min` 压到 `nt_samples`，因此 `partition` 恒为 1，不会出现 0。

## 5. mode 3: aggregate

### 三种聚合口径

官方 `compute_smoothing` 的三种口径，`smoothgrad_sq` 是 `E[X^2]` 而不是 `E[X]^2`，`vargrad` 是总体方差（除以 n，不做 Bessel 校正）：

```text
smoothgrad     = E[X]
smoothgrad_sq  = E[X^2]
vargrad        = E[X^2] - E[X]^2
```

### 跨 partition 累加

`update_sum_attribution_and_sq` 把每个 partition 的归因 `view(bsz, k, d)` 后 `sum(dim=1)` 与 `sum(X**2, dim=1)` 分开累加，最后统一乘以 `1 / nt_samples`：

```text
view(bsz, k, d).sum(dim=1)
expected_attributions    = sum_attributions    * 1 / nt_samples
expected_attributions_sq = sum_attributions_sq * 1 / nt_samples
```

注意平方是逐元素平方后求和，不是先求和再平方。

## 6. mode 4: baseline_expand

### draw 分支

`draw_baseline_from_distrib=True` 时按 `np.random.choice(num_ref_samples, n_samples * bsz)` 抽样索引，默认 `replace=True`，数量是 `n_samples * bsz` 而不是 `n_samples`：

```text
np.random.choice(num_ref_samples, n_samples * bsz)
```

### 默认分支

默认分支只在 `baseline.shape[0] == input.shape[0]` 且 `> 1` 时 `repeat_interleave(n_samples, dim=0)`，标量 / 单样本 baseline 原样保留。draw 分支的 `_validate_input` 只要求 `input.shape[1:] == baseline.shape[1:]`，不要求 batch 相同。

## 7. mode 5: noise_tunnel

### 完整编排

官方每个 partition 独立调用 `add_noise_to_inputs(k)`，因此整体 `perturbed` 是 `concat_p(repeat_interleave(inputs, k_p))`，**不是** 一次性 `repeat_interleave(inputs, nt_samples)`。每个 partition 的噪声偏移必须单独累加。

```text
perturbed = concat_p(repeat_interleave(inputs, k_p) + noise_p)
```

### delta 拼接

`return_convergence_delta=True` 时，官方实现是 `torch.cat(delta_partial_list, dim=0)`，而不是文档声称的「arithmetic mean across all nt_samples」：

```text
delta = torch.cat(delta_partial_list, dim=0)
```

## 8. 常见错误清单

1. 把 `repeat_interleave` 写成 `repeat` 的交替顺序。
2. `stdevs` 写成 `int 1` 而不是 `1.0`。
3. 以为 `stdevs` 传 tuple 可用（官方真 bug）。
4. `smoothgrad_sq` 算成 `E[X]^2`。
5. `vargrad` 用了样本方差的 `n-1` 校正。
6. `sum_sq` 算成 `(sum)^2`。
7. `partition` 用 `ceil` 而不是整除。
8. 忘记 `remaining` 单独成一批。
9. `bs > nt_samples` 没有用 `min` 压住。
10. 除以 partition 数而不是 `nt_samples`。
11. `draw` 抽样数写成 `n_samples` 而不是 `n_samples * bsz`。
12. 默认分支在任何 `num_ref == bsz` 时都扩展（漏掉 `> 1`）。
13. `delta` 做成算术平均而不是 `cat`。
14. 一次性 `repeat_interleave(inputs, nt_samples)` 构造 `perturbed`。
15. 第二个 partition 起用错噪声偏移。
16. 未知 `nt_type` 不报错。
17. `baseline_indices` 越界不报错。
18. `delta_parts` 数量与 partition 数不一致不报错。
19. `repeat_alt` 与 `repeat_interleave` 在 `bsz=1` 或 `k=1` 时天然相同，双向断言必须挑 `bsz>1` 且 `k>1` 的数据。
20. 域约束只断言「抛异常」而不看消息，删掉检查后让下游代为报错就能蒙混过关。

## 9. 自检

- `stdevs=0` 时应有 `smoothgrad = x`、`smoothgrad_sq = x^2`、`vargrad = 0`。
- 三种口径在方差非零数据上两两不同。
- `batch_sizes` 之和恒等于 `nt_samples`。
- `delta` 长度恒为 `bsz * nt_samples`。
- 每个 partition 的 `attributions` 按 `view(bsz, k, d)` 的 C 序索引，第 `bi` 个样本占 `part[bi*k:(bi+1)*k]`。
