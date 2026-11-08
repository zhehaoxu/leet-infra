# Captum GradientShap 期望梯度账：draw_baseline_from_distrib 的扁平 k*bsz 抽样、_scale_input 的 (n,1,1,..) alpha 广播、repeat_interleave(k)+normal 加噪布局、np.random 与 torch RNG 双流顺序、单 partition smoothgrad 聚合，以及 nt_samples_batch_size 不透出与 float32 噪声量化 quirk

## 1. 背景

Captum 的 `GradientShap` 把 Expected Gradients 近似成「基线分布抽样 + 随机插值点 + 加噪 + 梯度」的期望。本题把整条链路拆成五个可独立判题的数值账，全部用纯 Python 浮点实现，所有随机序列（基线索引 / alpha / 噪声）一律显式入参，不依赖 `captum` / `torch` / `numpy`，也不依赖任何全局 RNG。与 order 849（Kernel SHAP 采样）和 order 851（NoiseTunnel 平滑）不同，本题钉死的是 GradientShap 相对底层归因方法的硬编码口径、RNG 调用顺序与 delta 口径。

### 官方源码位置

- `captum/attr/_core/gradient_shap.py`：`GradientShap.attribute` / `InputBaselineXGradient.attribute` / `_scale_input`
- `captum/attr/_core/noise_tunnel.py`：`NoiseTunnel.attribute` / `add_noise_to_inputs`
- `captum/_utils/common.py`：`_expand_and_update_baselines` / `_format_callable_baseline`
- `captum/attr/_utils/attribution.py`：`GradientAttribution.compute_convergence_delta`

判题环境没有 `captum`，官方逐字执行环境靠注入 14 个 fake Captum modules、按 `common -> attr_common -> gradient -> attribution -> noise_tunnel -> gradient_shap` 顺序 `exec` 建立，探针跑的是真正的官方实现而不是重写版。

## 2. 口径约定

### 输入与输出

每个 tensor 是二维列表 `[bsz][d]`，tuple inputs 是 tensor 列表。`indices` 是扁平的 `k*bsz*n_in` 个基线索引，`alphas` 是 `k*bsz` 个 `[0,1)` 系数，`noises` 是 `k*bsz*d*n_in` 个噪声值。每个 mode 返回字典，字段名与官方变量对齐。所有非法输入统一抛 `ValueError`，消息点名被违反的约束。

### 数值精度与 float32 quirk

全程 Python float（float64）。官方 `add_noise_to_input` 用 `torch.tensor(stdev)` 构造标准差张量且不带 dtype，默认 float32，会把注入的 noise 量化出约 `1e-8` 误差。三层对拍时驱动临时 `torch.set_default_dtype(torch.float64)` 并在 finally 恢复，**官方源码本身不改**，使官方中间量与纯 Python reference 精确一致；参考解本身不受该 quirk 影响。

## 3. mode 1: baseline_draw

### 扁平 k*bsz 索引与有放回抽样

`GradientShap.attribute` 硬编码 `draw_baseline_from_distrib=True`，官方 `_expand_and_update_baselines` 走 `np.random.choice(num_ref, k*bsz, replace=True)`，得到 `k*bsz` 个**扁平**索引，而不是按 example 分组：

```text
indices = np.random.choice(num_ref, k*bsz, replace=True)
drawn   = [baselines[i] for i in indices]
layout  = flat_k_times_bsz
```

distrib 模式下 `_validate_input` 只校验 `shape[1:]`，不校验 batch 维，因此 `num_ref` 与 `bsz` 可以不等。`GradientShap` 断言 `baselines[0]` 必须是 Tensor；`_format_callable_baseline` 支持 0 参 / 1 参 callable、tensor 与 scalar。

### 非 distrib 的 repeat_interleave 与 applicable

`draw_baseline_from_distrib=False` 时官方走 `baseline.repeat_interleave(k, dim=0)`，但仅当 `num_ref == bsz 且 num_ref > 1` 时生效；否则原样保留。

```text
applicable = (num_ref == bsz and num_ref > 1)
applicable : layout = repeat_interleave, drawn_shape = [num_ref*k, d]
otherwise  : layout = unchanged,        drawn_shape = [num_ref, d]
```

### 域约束

`n_samples` / `bsz` 非正、索引长度不等于 `k*bsz`、索引非 int、索引越界、特征维不匹配都会抛 `ValueError`，消息分别点名对应约束。

## 4. mode 2: scale_input

### alpha 的 (n,1,1,..) 广播

`InputBaselineXGradient.attribute` 的 alpha 来自 `np.random.uniform(0.0, 1.0, bsz*k)`，在 `_scale_input` 里 view 成 `(n,1,1,..)` 后广播：

```text
inp_shape = (bsz,) + (1,) * (ndim - 1)
scaled    = alpha * input + (1.0 - alpha) * baseline
scaled    = scaled.requires_grad_()
```

注意 alpha 长度是 `bsz*k`（加噪后的 batch），插值两端是「加噪后的 input」与「抽出的 baseline」，不是原始 input。`requires_grad_()` 是官方显式调用。

### 域约束

baselines 行数 / 特征维必须与 inputs 一致，alphas 长度必须等于 batch 且落在 `[0,1)`，`ndim` 必须是 `>= 2` 的 int。

## 5. mode 3: noise_layout

### repeat_interleave(k) + normal

官方 `add_noise_to_inputs` 对每个输入 tensor 做 `input.repeat_interleave(k, dim=0) + torch.normal(0, stdev_expanded)`，噪声按 C order 的 `(i*k+j, t)` 铺开：

```text
stdev_expanded = torch.tensor(stdev).repeat((bsz*k,) + feature_shape)
noise          = torch.normal(0, stdev_expanded)
perturbed      = input.repeat_interleave(k, dim=0) + noise
```

`perturbed` 的 `bsz*k` 行按 `(i*k+j)` 排布，即 example i 的 k 份噪声连续，不是按噪声样本分组的 `repeat()` 布局。

### stdevs 校验

`stdevs` 必须是 Python `float` 标量，或长度等于输入 tensor 数的 tuple 且每个元素都是 `float`；`int` 会被官方断言拒绝。tuple 长度不匹配也抛异常。

### 域约束

`k` / `n_in` 必须为正，`noises` 长度必须等于 `bsz*k*d*n_in`，所有值必须是有限数字（bool 不算）。

## 6. mode 4: schedule

### 单 partition 与 RNG 双流

`GradientShap.attribute` **不透出 `nt_samples_batch_size`**，所以 `NoiseTunnel` 收到 `None`，恒为单 partition。RNG 调用顺序固定为 `choice -> normal -> uniform`：choice 与 uniform 共享 numpy 全局流，normal 走独立的 torch 流。

```text
order         = [choice]*n_in + [normal]*n_in + [uniform]
choice_calls  = [[num_ref, k*bsz, True] for each tensor]
normal_shapes = [[k*bsz, d] for each tensor]
uniform_calls = [[0.0, 1.0, k*bsz]]              # once, shared by all tensors
```

tuple inputs 时 choice / normal 每个 tensor 各一次，uniform 只有一次；单 tensor 时三者各一次。

### 直接 NoiseTunnel 的分片对照

若直接调用 `NoiseTunnel` 并给 `nt_samples_batch_size=b`，则会按 `p = k // b` 个满 partition 加一个余数 partition 分片，choice 只在开头调用一次（`kwargs_copy`），normal / uniform 每个 partition 各一次。本题用 `compare_batch_size` 显式给出这个对照账，但 GradientShap 本身永远 `partitions = 1`、`remaining = 0`、`batch_size_forwarded = False`。

## 7. mode 5: gradient_shap

### 端到端链路

```text
drawn     = baseline[indices[ti]]                 # choice, per tensor
perturbed = repeat_interleave(input, k) + noise   # normal, per tensor
scaled    = alpha*perturbed + (1-alpha)*drawn     # uniform, shared once
grads     = dF(scaled)                            # linear: w / quadratic: 2*w*scaled
attr      = (perturbed - drawn)*grads or grads    # multiply_by_inputs
out[i]    = sum_j attr[i*k+j] / k                 # single-partition smoothgrad mean
```

### smoothgrad 均值与 multiply_by_inputs

`nt_type` 被硬编码成 `smoothgrad`，聚合是 `view(bsz,k,..).sum(dim=1) * 1/nt_samples`，即按 example 分组求 `k` 份的均值，不是对全部 `bsz*k` 行求均值，也不是求和后不除 `k`。`multiply_by_inputs=True` 时 `InputBaselineXGradient` 用 `(input_noisy - baseline_drawn) * grad`，`False` 时直接返回 `grad`。

### convergence delta

`return_convergence_delta=True` 时 delta 形状是 `(bsz*k,)`，每个加噪样本一行：

```text
delta[r] = sum(attrs[r]) - (F(end) - F(start))
F(end)   = sum_t F(perturbed_t[r])
F(start) = sum_t F(drawn_t[r])
```

tuple inputs 时 `sum(attrs)` 跨所有 tensor 累加，而 `F(end) - F(start)` 只减一次（forward_func 接收全部 tensor）。

## 8. 官方 quirk 清单

1. `GradientShap.attribute` 硬编码 `nt_type="smoothgrad"` 与 `draw_baseline_from_distrib=True`，不透出 `nt_samples_batch_size`，永远单 partition。
2. 基线索引是扁平 `k*bsz` 个 `np.random.choice(..., replace=True)` 结果，不是按 example 分组。
3. distrib 模式只校验 `shape[1:]`，不校验 batch 维。
4. `_scale_input` 的插值两端是「加噪后的 input」与「抽出的 baseline」，alpha 长度是 `bsz*k`。
5. RNG 顺序是 choice(numpy) -> normal(torch) -> uniform(numpy)，choice 与 uniform 共享 numpy 流。
6. tuple inputs 时 choice / normal 每 tensor 一次，uniform 只有一次。
7. `torch.tensor(stdev)` 不带 dtype，默认 float32，注入噪声有约 `1e-8` 量化误差。
8. `stdevs` 必须是 float 或等长 float tuple，int 被拒绝。
9. `delta` 形状是 `(bsz*k,)`，不是 `(bsz,)`；跨 tensor 累加 attr 但只减一次 F 差。
10. `multiply_by_inputs=False` 时归因就是梯度本身，不带 `(input - baseline)` 因子。

## 9. 常见错误清单

1. 把基线索引按 example 分组，而不是扁平 `k*bsz`。
2. `replace` 写成 `False`，或漏掉「有放回」。
3. distrib 模式去校验 batch 维相等。
4. 非 distrib 的 `applicable` 漏掉 `num_ref > 1`。
5. `alpha.view` 写成 `(n, d)` 而不是 `(n,1,1,..)`。
6. 插值用了原始 input 而不是加噪后的 perturbed。
7. 忘了 `requires_grad_()`。
8. alpha 上界写成 `<= 1.0`。
9. 加噪用 `repeat(k)` 而不是 `repeat_interleave(k)`。
10. 噪声索引按 `(j*bsz+i)` 分组而不是 `(i*k+j)` 的 C order。
11. `stdevs` 接受 int 或 tuple 元素为 int。
12. `noises` 长度按 `bsz*k*d` 算，漏乘 `n_in`。
13. 把 `nt_samples_batch_size` 当成可透出参数，产生多个 partition。
14. RNG 顺序写成 uniform 先于 choice，或 tuple 时 uniform 也每 tensor 一次。
15. smoothgrad 均值对全部 `bsz*k` 行求均值，或求和后不除 `k`。
16. `multiply_by_inputs` 分支写反：False 时仍乘 `(input - baseline)`。
17. 二次型梯度写成 `w*scaled` 而不是 `2*w*scaled`。
18. `delta` 形状写成 `(bsz,)`，或 tuple 时每个 tensor 各减一次 F 差。
19. `delta` 符号写成 `(F(end)-F(start)) - sum(attrs)`。
20. 报告字段 `partition_count` / `n_in` / `feature_dim` 被改错。
21. `return_convergence_delta=False` 时仍返回 delta。
22. 未知 mode 不抛 `unknown mode`。
23. 域约束只断言「抛异常」而不看消息关键字。
24. 数值校验把 bool 当成有限数字接受。
25. 常数噪声序列导致多 tensor 切片与布局差异不可观测。

## 10. 自检

- `num_ref == bsz == 1` 时非 distrib 仍为 `unchanged`，`drawn_shape` 保持 `[num_ref, d]`。
- tuple inputs 时 `per_tensor` 两段 noise / indices 切片互不相同。
- `bsz > 1 且 k > 1` 时 `(i*k+j)` 与 `(j*bsz+i)` 布局必须不同。
- `return_convergence_delta` 开关必须精确控制 `delta` 字段有无。
- 三层对拍、判题用例、mutation 全部通过后才落地。
