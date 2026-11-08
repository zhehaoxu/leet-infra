# Captum FeatureAblation 分组消融账：max+1 与 max-min+1 双口径、ablations_per_eval 分批与尾批重展开、初始/扰动 eval 差值累加、use_weights 权重除法与整条 attribute 编排

## 1. 背景

本题重建 `captum.attr.FeatureAblation` 的**分组消融数值账**，而不是调用库函数。判题环境没有 `captum`，也不允许依赖 `torch`；所有 mode 必须是纯 Python 的确定性数值账，输入是嵌套列表，输出是题目约定的复合字典。FeatureAblation 的核心不是「滑掉连续窗口」，而是：每个标量位置带一个整数 `feature_mask` 编号，编号相同的所有位置构成一组，一次前向里同时把这组替换成 baseline；用 `initial_eval - modified_eval` 得到这一组的贡献；分批参数 `perturbations_per_eval`（下称 `ppe`）把若干组拼进同一个 batch。

### 官方源码位置

回源目标是 `meta-pytorch/captum main::captum/attr/_core/feature_ablation.py`（1075 行 / 47472 bytes；owner 已迁到 `meta-pytorch`）。另外复用同仓 `captum/_utils/common.py` 与 `captum/attr/_utils/common.py`（854 已回源），提供 `_format_feature_mask` / `_format_input_baseline` / `_expand_target` / `_expand_additional_forward_args` 等官方 helper。本题只取与数值账有关的方法：`_get_feature_range_and_mask`、`_get_feature_counts`、`_attribute_progress_setup`、`_construct_ablated_input`、`_ith_input_ablation_generator`、`_process_initial_eval`、`_process_ablated_out`、`_accumulate_for_single_input`、`_generate_result`，以及 `attribute` 的编排。

### 逐字执行环境

三个官方 `.py` 文件整文件 `exec` 到同一个命名空间，官方源码一行不改；只 fake 掉确实不可用的依赖：typing 别名、`log_usage` 装饰器、`progress` 进度条、`Attribution` / `PerturbationAttribution` / `GradientAttribution` 三个 ABC、`SkLearnLasso` 与 `FeatureAblationFutureError`。`torch.futures.Future` / `collect_all` 是真实的。驱动层把嵌套列表转成 `float64` 张量、`long` mask，用确定性的 `SpyForward`（线性 / 二次 / 多输出 / 聚合 / Python 标量）记录每一次前向的入参形状与数值。

## 2. 口径约定

### 输入与输出

单入口是 `solve(mode, **kw)`，五个 mode 的必填参数与默认值如下；`inputs` 永远是「每个元素是一份输入张量」的列表，第 0 维恒为 `num_examples`。

```text
solve("mask_range",  input, feature_mask=None, offset=0,
                     perturbations_per_eval=1)
solve("ablate",      expanded_input, input_mask, baseline=0,
                     start_feature, end_feature)
solve("schedule",    inputs, feature_mask=None, baselines=None,
                     index=0, perturbations_per_eval=1,
                     target=None, additional_forward_args=None,
                     extra_args={})
solve("eval_out",    initial_eval, input_shapes, modified_evals,
                     masks, batch_sizes, num_examples,
                     perturbations_per_eval=1, index=0,
                     use_weights=False, initial_dtype="float64")
solve("attribution", inputs, forward_weights, feature_mask=None,
                     baselines=None, tuple_inputs=False,
                     perturbations_per_eval=1, forward_kind="linear",
                     target=None, n_classes=2, use_weights=False,
                     show_progress=False, additional_forward_args=None,
                     extra_args={})
```

### 数值精度

全部用 Python `int` / `float`（IEEE double），嵌套列表按 C 序展平与回填，广播规则与 NumPy 一致：从尾部对齐，每个维度必须相等或其中之一为 1。不使用全局随机源，所有扰动、baseline、mask、forward 权重显式入参。未知 mode 抛 `ValueError` 且消息含 `unknown mode`；所有非法输入抛 `ValueError`，消息点名被违反的约束。

## 3. mode 1: mask_range

### 双口径

官方有**两个不同的计数**，绝不能混为一谈：

```text
mn            = min(flatten(feature_mask))
mx            = max(flatten(feature_mask))
num_features  = mx + 1                  # 循环上界、ppe clamp 的分母
feature_count = mx - mn + 1            # _get_feature_counts，进度条用
loop_start    = mn                     # 从最小编号开始，不是 0
num_steps     = num_features - mn
ppe_clamped   = min(ppe, num_features) # 注意用 max+1，不是 max-min+1
chunks        = [[mn, mn+ppe_clamped], ...] 直到 num_features
progress_total= ceil(feature_count / ppe) + 1
```

`feature_mask=None` 时官方为每个输入生成默认 mask：形状 `[1] + input.shape[1:]`，值是从 `offset` 起的连续整数；多输入时 `offset` 全局累加（第 2 份输入的编号接着第 1 份）。显式 mask 必须与输入同 rank、第 0 维为 1 或 `num_examples`、其余各维为 1 或对应输入维，且值为非负整数。尾维为 1 的 mask 合法（广播）。

### 域约束

`input` 至少 2 维（有 leading example 维）；`offset >= 0`；`ppe` 必须是 `>= 1` 的整数，`bool` 按官方 `isinstance(True, int)` quirk 视为 1。mask rank 不符抛 `feature mask must have the same rank as the input`，首维不符抛 `feature mask first dim must be 1 or num_examples`，其余维不符抛 `feature mask shape is not broadcastable to input`，值非法（float / bool / 负数）抛 `feature mask values must be non-negative integers`。

## 4. mode 2: ablate

### 掩码堆叠与混合

对应官方 `_construct_ablated_input(expanded_input, input_mask, baseline, start_feature, end_feature)`。`expanded_input` 形状是 `(K, num_examples, ...)`，其中 `K = end_feature - start_feature` 是这一步同时消融的组数。对每个 `j in range(start, end)` 生成一张 0/1 掩码：

```text
mask[j - start] = (input_mask == j) ? 1 : 0
keep            = 1 - mask
ablated        = expanded_input * keep + baseline * mask
```

`start_feature > 0` 时若 `input_mask` 里没有对应编号，该张掩码全 0，输入原样保留。`baseline` 可以是标量，也可以是与掩码广播兼容的张量（例如逐行不同的 baseline）。返回 `mask` / `mask_shape` / `ablated` / `ablated_shape` / `num_ablated_features` / `ones_per_feature`（每张掩码里 1 的个数）/ `baseline_shape`。

### 广播与域约束

掩码堆叠后的形状 `[K] + input_mask.shape` 必须能广播到 `expanded_input.shape`，且广播结果**不得改变** `expanded_input` 的形状；后者不满足时抛 `input_mask broadcast changes the expanded_input shape`。`input_mask` 必须是整数；`start_feature` / `end_feature` 必须是整数且 `0 <= start < end`。

## 5. mode 3: schedule

### 分批重复与尾批

对应 `_ith_input_ablation_generator`。设 `N = num_examples`、`ppe_eff = min(ppe, num_features)`（同样用 `max+1`）。`ppe_eff > 1` 时**所有**输入都按行重复 `ppe_eff` 次（`[row for _ in range(k) for row in x]`，即整份输入重复，不是逐元素 repeat），`target` 与张量型 `additional_forward_args` 同步展开；标量型附加参数保持不变。循环从 `mn` 起，每批 `K = min(ppe_eff, num_features - pos)`：

```text
repeated_batch = [repeat_rows(x, ppe_eff) for x in inputs]   # 当 ppe_eff>1
cur[i]         = reshape(cur[i], [K, N] + feat_shape)
cur_mask       = [ (mask_i == j) for j in range(pos, pos+K) ]
cur[i]         = cur[i] * (1-cur_mask) + baseline * cur_mask
cur[i]         = reshape(cur[i], [K*N] + feat_shape)
batch_size     = K * N
```

★ **尾批 quirk**：当 `K != ppe_eff` 时，官方不是简单切掉重复张量的尾部，而是从**原始** `inputs` / `target` / `additional_forward_args` 重新按 `K` 展开一遍，然后对**每一个**输入（不只是被消融的那一份）切前 `K*N` 行。因此尾批的张量附加参数长度是 `K`，不是 `ppe_eff`；其它输入的尾部也会被切掉。满批时沿用已展开的版本。

### 参数展开与 kwargs

`target` 是 list 时按 `list(target) * k` 展开，是标量时原样保留。`additional_forward_args` 里 list（张量）元素按行重复 `k` 次，标量元素不动。`extra_args` 里的 **tuple** 值按下标 `index` 取值，非 tuple 值原样传入；返回值 `extra_args_selected` 记录这个选择结果。张量 baseline 会被 reshape 成 `[1] + baseline.shape`，标量 baseline 的 `baseline_reshape` 是 `[]`。进度 `progress_total = sum(ceil(_mask_count(m) / ppe) for m in masks) + 1`，其中 `_mask_count = max-min+1`。

### 域约束

`inputs` 非空；`feature_mask` / `baselines` 的数量必须等于 `inputs` 数；baseline 首维为 1 或 `num_examples`；`additional_forward_args` 必须是 list；`index` 在范围内；`ppe` 校验同 mode 1。

## 6. mode 4: eval_out

### 初始状态

对应 `_process_initial_eval`：Python `int` 的 `attrib_type` 是 `int64`、Python `float` 与张量是 `float64`；`n_outputs = numel(initial_shape)`，标量时是 1；`flattened_initial_eval` 形状恒为 `[1, n_outputs]`。累加器 `total[i]` 形状 `[n_outputs] + input_shapes[i][1:]`、初值 0；`use_weights=True` 时额外分配同形状的 `weights`，否则为空列表。

### 差值累加与权重除法

对应 `_process_ablated_out` / `_generate_result`。对每条记录：

```text
diff      = initial_eval_flat[None, :] - reshape(modified_eval, (-1, n_outputs))
diff      = reshape(diff, shape(diff) + [1]*(input_rank-1))   # 广播到特征维
total[i] += sum_axis0(diff * mask)
weights[i] += sum_axis0(float(mask))                 # 仅 use_weights
attrib    = total / weights                          # 仅 use_weights，除零得 nan
```

`ppe > 1` 时第一条记录要断言 `modified_eval` 的第 0 维等于 `(batch_size / num_examples) * initial_shape[0]`，否则抛 `output should be a tensor whose 1st dim grow`（官方要求聚合输出随 batch 增长）。`modified_eval` 展平长度必须能被 `n_outputs` 整除，否则抛 `modified eval cannot reshape to (-1, n_outputs)`。返回 `attributions` / `weights` / `flattened_initial_eval` / `attrib_type` / `n_outputs` / `total_attrib_shapes` / `eval_diffs` / `initial_shape`。

### 域约束

`input_shapes` 非空，`index` 在范围内；`modified_evals` / `masks` / `batch_sizes` 三者长度必须相等；`initial_eval` 不能是 `bool`。空记录时累加器形状与初始形状原样返回、attributions 全 0。

## 7. mode 5: attribution

### 五种 forward_kind

本题用显式权重的确定性前向替代真模型，六种 `forward_kind`：`linear`（`sum(w_j * x_j)`）、`quadratic`（`sum(w_j * x_j²)`）、`multi`（每个 batch 输出 `n_classes` 个 `sum(w[c][j] * x[j])`，再按 `target` 选取）、`aggregate`（整批求和返回 Python float）、`python_int` / `python_float`（同上但强制转型，用于验 dtype 传播）。`additional_forward_args` 在 attribution 里只允许标量数值，累加成一个 bias。

### 端到端编排

先跑一次原始输入得到 `initial_eval`，然后按输入逐个、按 chunk 逐个跑扰动前向，累加差值，最后按 `use_weights` 决定是否除以权重。返回 `attributions`（`tuple_inputs=False` 时只取第 0 份）/ `weights` / `n_outputs` / `attrib_type` / `flattened_initial_eval` / `initial_shape` / `num_forwards` / `forward_shapes` / `chunks`（含 `input_index` / `start` / `end` / `batch_size`）/ `progress_total`（仅 `show_progress=True`，否则 `None`）。★ 官方会**整份跳过** `numel == 0` 的输入（空张量），其 attributions 为空且不计入 forward 次数。多输入时默认 mask 的编号全局连续，所以第 2 份输入的 chunk 从前面累计的编号开始。

### 域约束

单输入模式 `inputs` 必须恰好一份，多输入需 `tuple_inputs=True`；`forward_kind` 未知抛 `unknown forward_kind`；`multi` 的 `target` 必须是 int 或与 batch 等长的 list，否则抛 `Target list length does not match output!` / `multi forward needs an integer or list target`。

## 8. 官方 quirk 清单

1. `num_features = max+1`（循环上界 / ppe clamp）与 `feature_count = max-min+1`（进度）是两套口径；`min>0` 时两者不同。
2. `isinstance(True, int)` 为真，`ppe=True` 被官方当作 1 接受。
3. 尾批（未满 `ppe_eff`）从原始输入重新展开，且切的是**所有**输入的 `K*N` 行，不只是被消融的那份。
4. `extra_args` 中的 tuple 按 `index` 取值，非 tuple 原样传。
5. `ppe>1` 时对聚合标量前向抛断言，要求输出第 0 维随 batch 增长。
6. `use_weights` 时权重是 mask 的浮点累加，除零得 `nan`。
7. `numel == 0` 的输入被整份跳过。
8. Python `int` / `float` 前向的 dtype 会传播到 `attrib_type`（`int64` / `float64`）。

## 9. 常见错误清单

- 把 `ppe_clamped` 写成 `min(ppe, max-min+1)`，`min>0` 的 mask 会错。
- 默认 mask 的多输入编号从 0 重启，没有全局累加。
- 把 `repeat_rows` 写成逐元素 repeat，batch 内行序会错。
- 尾批沿用已展开的 `ppe_eff` 份输入，或对张量附加参数按 `ppe_eff` 而非 `K` 展开。
- 把 `diff` 直接乘 mask 而忘掉补 `[1]*(input_rank-1)` 的广播维。
- 用 `sum_axis0` 时把扰动行也累加进去（应只在 chunk 内求和，再累加到 total）。
- 把 `show_progress=False` 时也返回非 `None` 的 `progress_total`。
- 忘记跳过空输入，或把 `tuple_inputs` 的返回结构拍平。
- 多输入时 `chunks` 的 `start/end` 用了局部编号而非全局编号。

## 10. 自检

判题用例覆盖五条链路的公式、双口径、尾批、广播、权重除法、dtype、进度与 quirk；参考解需通过 12 段官方探针（87 项）、三层对拍（official / reference / 独立 oracle）与全量 mutation。任何未知 mode 必须抛 `ValueError`，消息包含 `unknown mode`。
