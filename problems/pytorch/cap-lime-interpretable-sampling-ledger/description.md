# Captum Lime 可解释采样账：Bernoulli 二值掩码与 feature_mask 索引映射、指数核的 2*kw² 分母、batch_count 恒等于 n_samples 的 DataLoader 口径、加权岭回归代理模型与 coefs[mask] 回填，以及默认 n_samples=25 与 float32 回填 quirk

## 1. 背景

Captum 的 `Lime` 用「可解释空间二值采样 -> 映射回输入空间 -> 前向打分 -> 相似度加权 -> 训练线性代理模型 -> 把系数回填到输入形状」来解释单个样本。本题把整条链路拆成五个可独立判题的数值账，全部用纯 Python 实现，不依赖 `captum` / `torch` / `sklearn` / `numpy`，也不依赖任何全局 RNG：Bernoulli 采样结果 `samples` 与可选核权重 `weights` 一律显式入参。与 order 849（Kernel SHAP 采样）、order 851（NoiseTunnel 平滑）、order 853（GradientShap 期望梯度）不同，本题钉死的是 Lime 的**扰动式代理模型**口径：二值可解释特征、feature_mask 分组、指数核、`perturbations_per_eval` 分批与系数回填。

### 官方源码位置

```text
captum/attr/_core/lime.py                     LimeBase / Lime / default_* / construct_feature_mask
captum/_utils/common.py                       _run_forward / _flatten_tensor_or_tuple / _reduce_list
captum/attr/_utils/common.py                  _construct_default_feature_mask / _format_input_baseline
captum/attr/_utils/batching.py                _batch_example_iterator
captum/attr/_utils/attribution.py             PerturbationAttribution
captum/_utils/models/linear_model/model.py    SkLearnLasso
```

仓库 owner 已从 `pytorch/captum` 迁到 `meta-pytorch/captum`；官方主源码 1295 行。

### 逐字执行环境

判题环境没有 `captum` 与 `sklearn`。三层对拍的 official 层把 `common.py` / `attr_common.py` / `batching.py` / `lime.py` 逐字 `exec` 进共享命名空间，只 fake 掉确实不可用的依赖（typing / progress / log_usage / attribution 基类 / models / approximation_methods），并注入确定性的 `torch.bernoulli` 序列与显式的加权岭回归替身 `RidgeModel`。探针跑的是真正的官方实现，不是重写版。

## 2. 口径约定

### 输入与输出

每个 tensor 是二维列表 `[1][d]`（只支持 `bsz=1`），tuple inputs 是 tensor 列表并用 `tuple_inputs=True` 标记。`samples` 是 `n_samples` 行、每行 `F` 个 int 0/1 的二值矩阵；`weights` 若给出则是 `n_samples` 个非负核权重，省略时由 `kernel` 公式现算。每个 mode 返回字典，字段名与官方变量或账本字段对齐。所有非法输入统一抛 `ValueError`，消息点名被违反的约束；未知 mode 抛 `unknown mode`。

### 数值精度与 float32 quirk

参考解全程 Python float（float64），代理模型用 `Fraction` 精确解线性方程组。官方侧有两处 float32 口径：核函数把输入 `.float()` 后在 float32 张量上算 `distance**2` 再 `math.exp`，与纯 double 复算差约 `1e-9`；`_convert_output_shape` 在 `torch.zeros_like(..., dtype=torch.float)` 上累加 `coefs[f].item() * (mask == f).float()`，回填值与 double 解差约 `1e-7`。三层对拍对这两处使用对应容差，参考解本身不受影响。

## 3. mode 1: feature_mask

### 默认 mask 与显式 mask

`construct_feature_mask(None, formatted_inputs)` 走 `_construct_default_feature_mask`：按 tensor 顺序给每个标量一个**跨 tensor 累加**的索引，特征数 `F = sum(numel(t[0]))`。显式 mask 先 tuple 化，取所有张量的全局最小值 `min_index`；若 `min_index != 0` 则**整体减去该最小值**（并置 `shifted=True`），最后 `F = max(mask) + 1`。注意 `F` 由最大值决定，索引可以不连续。

```text
default : mask[t] = [idx .. idx+d-1], idx += d ; F = idx
explicit: if min != 0: mask -= min
          F = max(mask) + 1
          shifted = (min != 0), default = False
```

### 域约束

`inputs` 必须非空、每个 tensor 恰好一行、数值有限；`feature_mask` 的索引必须是非负 int、每行长度与对应输入一致、tensor 数与输入一致；`tuple_inputs` 必须与实际 tensor 数匹配。

## 4. mode 2: interp_transform

### 二值向量到输入空间

官方 `default_perturb_func` 返回形状 `(1, F)` 的 `long` 二值向量：`torch.bernoulli(torch.ones(1, F) * 0.5)`。`default_from_interp_rep_transform` 用 `curr_sample[0][feature_mask].bool()` 把该向量按 mask **索引映射**回输入形状，再逐元素混合：

```text
binary_mask = sample[feature_mask].bool()
perturbed   = binary_mask * original + (~binary_mask) * baselines
F           = num_interp_features
sample[i] in {0, 1} : 1 keeps original, 0 uses baseline
```

tuple 分支对每个 tensor 各取一次 `feature_mask[j]`；不同 tensor 使用同一个索引即被归为同一可解释特征，会同时开关。

### 域约束

`sample` 长度必须等于 `F`、元素必须是 int 0/1（bool 与非 int 均被拒绝）；baselines 的 tensor 数与特征维必须与 inputs 一致。

## 5. mode 3: kernel

### 指数核公式

`get_exp_kernel_similarity_function(distance_mode, kernel_width)` 返回的闭包先把原始与扰动输入展平并转 float32。`cosine` 用 `1 - CosineSimilarity(dim=0)(a, b)`（分母 `max(||a||*||b||, 1e-8)`），`euclidean` 用 `torch.norm(a - b)`；随后统一过指数核：

```text
distance(cosine)    = 1 - dot(a,b) / max(|a|*|b|, 1e-8)
distance(euclidean) = sqrt(sum((a[i]-b[i])^2))
weight              = exp( -(distance^2) / (2 * kernel_width^2) )
```

分母是 `2 * kernel_width**2` **不是** `kernel_width**2`；默认 `distance_mode="cosine"`、`kernel_width=1.0`。

### 域约束

`distance_mode` 只能是 `cosine` / `euclidean`，`kernel_width` 必须是正的有限数（bool 不算），`original` 与 `perturbed` 展平后长度必须相同。

## 6. mode 4: surrogate

### 加权岭回归

官方默认代理模型是 `SkLearnLasso(alpha=0.01)`（依赖 sklearn）。本题按通用纪律把上游依赖剥除，改为可精确判题的**加权岭回归**：对 `n x F` 二值设计矩阵 `X`、输出 `y`、权重 `w`，

```text
gram[a][b] = sum_i w[i] * X[i][a] * X[i][b]   (a,b = 0..F-1)
gram[a][a] += alpha
rhs[a]     = sum_i w[i] * X[i][a] * y[i]
coefs      = solve(gram, rhs)
```

`alpha=0` 时退化为加权最小二乘；`alpha>0` 时系数被收缩。奇异 Gram 抛 `ValueError`。

### 域约束

`interp_samples` 非空、每行等长且元素为 int 0/1；`outputs` / `weights` 长度等于样本数、均为有限数值且权重非负；`alpha` 非负。

## 7. mode 5: lime

### 端到端链路

逐个 sample 走 `interp_transform` 得到扰动输入，按 `forward_kind` 前向打分（`linear`: `sum(w[j]*x[j])`；`quadratic`: `sum(w[j]*x[j]^2)`，tuple 时各 tensor 相加），再取显式 `weights[i]` 或由 `kernel` 现算权重；最后把 `samples / outputs / weights` 交给 `surrogate` 求系数。返回值同时给出 `perturbed` / `binary_masks` / `gram` / `rhs` 等全部中间量。

### 分批与 DataLoader

官方 `LimeBase.attribute` 每累计 `perturbations_per_eval` 个样本调用一次 `_evaluate_batch`，不足一批的尾部单独再调一次；`batch_count` 在每个样本处 `+= 1` 且**从不重置**，因此 `DataLoader(dataset, batch_size=batch_count)` 的 batch_size 恒等于 `n_samples`，代理模型一次看到全部样本。

```text
batches       = [ppe] * (n // ppe) + ([n % ppe] if n % ppe else [])
batch_count   = n_samples            (never reset)
loader_batch_size = n_samples
dataset_shapes    = [[n, F], [n], [n]]
```

### 系数回填

`return_input_shape=True` 时按 mask 把系数铺回输入形状：`attr[t][j] = sum_f coefs[f] * (mask[t][j] == f)`，同一 mask 值共享同一系数；`False` 时直接返回长度为 `F` 的一维 `coefs`。

```text
attr[t][j] = sum(coefs[f] for f in range(F) if mask[t][j] == f)
return_input_shape=False -> attribution = coefs
```

### 官方 quirk 清单

`Lime.attribute` 默认 `n_samples=25`，但同一段 docstring 写 `Default: 50`；`LimeBase.attribute` 默认才是 50。`perturb_interpretable_space=True` 且 `to_interp_rep_transform=None`。`bsz>1` 且前向逐样本输出时 `_attribute_kwargs` 为每个 example 单独训练一个代理模型并 `_reduce_list`；若前向返回单 batch 标量则断言 `perturbations_per_eval == 1`。`has_convergence_delta()` 与 `multiplies_by_inputs` 均为 `False`。

## 8. 常见错误清单

- 把 `feature_mask` 当成布尔掩码而不是**索引数组**：官方用 `sample[feature_mask]` 取值，不是逐元素 `sample & mask`。
- 忘记 `min_index != 0` 的整体平移，或把 `F` 算成 `len(unique(mask))` 而不是 `max + 1`。
- 把指数核分母写成 `kernel_width**2`，或把 cosine 距离写成 `cos` 本身。
- 用 `repeat` 思路重排样本、把 `batch_count` 在每批后清零，或让 DataLoader 的 batch_size 等于 `perturbations_per_eval`。
- 训练代理模型时漏掉核权重、把 `alpha` 加到所有矩阵元素而不是对角线，或用未正则化的普通最小二乘替代岭回归。
- `return_input_shape=True` 时把系数按位置顺序平铺，而不是按 mask 索引分组回填。
- 让 `samples` 含 bool / 浮点、让 `weights` 含负数，或用全局随机源现采 Bernoulli。

## 9. 自检

- 五个 mode 均可通过 `solve(mode, **kw)` 独立调用；`unknown mode` 抛 `ValueError`。
- 判题用例 8 个 `check_*(solve)`，参考解 8/8；三层对拍 374 检查项 ALL GREEN；mutation 156/156 全 CAUGHT。
- 中英题面 heading 数与层级一致、fenced code block 数与内容一致；`meta.json` 含 `title_en`；starter / solution / test_cases 均可编译。
