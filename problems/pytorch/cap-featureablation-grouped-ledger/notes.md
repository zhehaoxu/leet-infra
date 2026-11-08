# 856 解答笔记：Captum FeatureAblation 分组消融账

## 1. 官方回源

### 源码清单与依赖

主源码是 `meta-pytorch/captum main::captum/attr/_core/feature_ablation.py`（1075 行 / 47472 bytes）。Captum 的 owner 已迁到 `meta-pytorch`，旧 `pytorch/captum` 的 raw URL 会 404，改用 `api.github.com/repos/meta-pytorch/captum/contents/...?ref=main` 取 base64。另复用 854 已回源的 `captum/_utils/common.py` 与 `captum/attr/_utils/common.py`。原候选 Integrated Gradients 与 DeepLift 因全库已有 `int-captum-integrated-gradients-ledger` 与 `cap-deeplift-nonlinear-rule-ledger` 而弃用。

### 逐字执行环境

三个官方文件整文件 `exec` 到共享 `NS`，只 fake typing 别名、`log_usage`、`progress`、三个 ABC、`SkLearnLasso`、`FeatureAblationFutureError`；`torch.futures.Future` / `collect_all` 真实可用。每次 exec 后把 `NS` 同步进 `captum._utils.common` 等模块对象，解决官方模块间的 `from ... import ...` 循环。

```text
for name, folder in (("common.py", D854), ("attr_common.py", D854),
                     ("feature_ablation.py", D856)):
    exec(compile((folder / name).read_text(), name, "exec"), NS)
    for mod in (common, attr_common, batching):
        for k, v in NS.items():
            if not k.startswith("__"):
                setattr(mod, k, v)
```

驱动层用 `SpyForward` 记录每次前向的 `shapes` / `values` / `extras`，提供 `comp_mask_range` / `comp_construct` / `comp_generator` / `comp_range` / `comp_progress` / `comp_process_initial` / `comp_process_ablated` / `run_attribute` 八类对照入口，全部输出转成嵌套列表。官方源码一行不改，对拍统一在 `float64` 下完成。

## 2. 十二段官方机制探针

### 探针结论

`probe856.py` 十二段共 87 项断言全绿：P1 默认 mask（单输入 / 3D / tuple 全局偏移）；P2 `min>0` 时 `max+1` 与 `max-min+1` 分歧及 gap mask；P3 `_construct_ablated_input` 的 0/1 堆叠、标量/张量 baseline、越界 start 的全零掩码；P4 `ppe=1` 只改第 index 份输入；P5 `ppe=3` 全输入重复、张量附加参数按行 cat、尾批 target 按 K 重展开；P6 `min>0` + 大 ppe 的空洞掩码与 `K*N` batch；P7 tuple kwargs 按 index 取值与两个 expand helper；P8 `_process_initial_eval` 的 dtype / `n_outputs` / 展平 eval / 累加器形状 / `use_weights` 分配；P9 `_process_ablated_out` 的差值、mask 广播、sum over dim 0 与权重除法；P10 `ppe` 校验、`bool` quirk、聚合输出断言、Python int 输出；P11 forward 次数、扩展 batch 形状、进度条 total/updates/closed；P12 五种 forward_kind、target 选择、baseline、use_weights、5 特征 ppe=3 的尾批。

★ 首轮 1 项 FAIL：P12 把标量 baseline 的输出写成 `[[-2., -2.], [-2., -2.]]`，实际官方是 `[[-1., 0.], [1., 2.]]`（baseline=2 只替换被消融的位置，未被消融的原值保留），修正期望后全绿。

## 3. 三层对拍

### official 层

逐字执行的官方类，覆盖 5 组 mask_range、6 组 ablate、10 组 schedule、7 组 eval_out、20 余组 attribution，含 3D 输入、tuple 输入、min>0 mask、张量 baseline、ppe=2/3/5、全部 forward_kind、use_weights、show_progress、空输入。

### reference 层

`solution_staged.py` 五个 mode，纯 Python 无 torch；`shape_of` / `flatten` / `unflatten` / `broadcast_to` / `bop` / `sum_axis0` 负责任意维度展平、广播与逐元素运算；`_mask_eq` / `_expand_add` / `_expand_target` / `_mask_count` 复刻官方 helper 语义。

### oracle 层

独立实现：mask 范围用朴素 `min/max` 与显式 chunk 循环；ablate 用逐位置 `if mask[p] == j: baseline else expanded`；schedule 用「为每个 j 单独构造一份完整输入」的朴素生成器；eval_out 用显式 `(init[j] - mod[j]) * mask[p]` 累加；attribution 用「每组特征单独跑一次完整前向」的朴素定义式，与官方分批实现不同源。

### 对拍结论

```text
VERIFY 856 DONE checks=5291 fails=0 ALL GREEN
```

三层互不相同，避免同源错误。首轮失败集中在把 generator 的 yield 顺序误当成 `attribute` 内部的批顺序，以及独立 oracle 把 `[freq][time]` 式的行序写反；修正后 5291 项全绿。

## 4. 官方 quirk 清单

### 双口径与 bool ppe

`_get_feature_range_and_mask` 返回 `num_features = max+1`、`min_feature = min`；`_get_feature_counts` 返回 `max-min+1`。`ppe_eff = min(ppe, max+1)`，但进度 total 用 `ceil((max-min+1)/ppe)+1`。`min>0` 时两者不同，是最容易写错的地方。`assert isinstance(ppe, int)` 使 `True` 通过校验并当作 1。

### 尾批重展开

`_ith_input_ablation_generator` 在 `K != ppe_eff` 时从**未重复**的原始 `inputs` / `target` / `additional_forward_args` 重新按 K 展开，并对所有输入切 `[0:K*N]`。因此尾批的张量附加参数长度是 K，其它输入也会被裁到 K*N 行；满批则沿用已展开版本。

### 空输入与标量前向

`attribute` 对 `numel(inputs[i]) == 0` 的输入 `continue`，其 attributions 为空且不计入 forward 次数。Python `int` / `float` 前向经过 `_process_initial_eval` 后 `attrib_type` 分别是 `int64` / `float64`；`aggregate` 类标量输出在 `ppe>1` 时因第 0 维不随 batch 增长而触发官方断言。

## 5. 五个 mode 的实现要点

### mask_range

`_ppe` 统一校验（bool 特判为 `int(p)`）；`_validate_mask` 检查 rank / 首维 / 各维 / 值类型；默认 mask 用 `offset` 起连续整数，形状 `[1] + ishape[1:]`。返回 13 个字段，`chunks` 是 `[start, end)` 列表，`progress_total = ceil(feature_count/ppe)+1`。

### ablate

先校验 `expanded_input` rank 与 `start/end`，再算 `[K] + mshape` 与 `eshape` 的广播，`merged != eshape` 时抛专属消息。`current_mask = [_mask_eq(mask, j) ...]`，`keep = 1 - mask`，用 `bop` 做两次广播乘加；`ones_per_feature` 逐张计数。

### schedule

`_generate` 先算 `ppe_eff`，`ppe_eff > 1` 时对所有输入 `repeat_rows`，baseline 张量 reshape 成 `[1]+shape`。循环里 `K = min(ppe_eff, num_features - pos)`，`K != ppe_eff` 时重新展开并切所有输入；`cur[index]` 先 reshape 成 `[K, N] + feat_shape` 再与 mask 混合，最后展回 `[K*N] + feat_shape`。`extra_args` 的 tuple 按 index 取值。

### eval_out

`_initial_state` 处理 int / float / 数组三种 initial，构造 `[1, n_outputs]` 的 flattened 与 `[n_outputs]+s[1:]` 累加器。`_accumulate` 逐记录做 `ppe>1` 形状断言、`(-1, n_outputs)` reshape、diff 补广播维、`sum_axis0(diff*mask)` 累加、`use_weights` 时累加浮点 mask。`_generate_result` 按权重相除，除零得 `nan`。

### attribution

六种 forward_kind 共用 `_forward`；初始前向 + 逐输入逐 chunk 的扰动前向，累积 `forward_shapes` 与 `chunks`。空输入 `numel == 0` 跳过。`show_progress` 时才算 progress_total。`tuple_inputs` 决定返回全部 attributions 还是只取第 0 份。

## 6. 域约束与错误消息

所有非法输入抛 `ValueError` 并点名约束：`perturbations_per_eval must be an integer and at least 1`、`input must have a leading example dimension`、`offset must be non-negative`、`feature mask must have the same rank as the input`、`feature mask first dim must be 1 or num_examples`、`feature mask shape is not broadcastable to input`、`feature mask values must be non-negative integers`、`expanded_input must have shape (n_features, n_examples, ...)`、`input_mask values must be integers`、`start_feature and end_feature must be integers`、`start_feature must be less than end_feature`、`start_feature must be non-negative`、`input_mask is not broadcastable to expanded_input`、`input_mask broadcast changes the expanded_input shape`、`inputs must be a non-empty list of input arrays`、`feature mask count must match inputs`、`baselines count must match inputs`、`baseline must be a number or a broadcastable array`、`baseline first dim must be 1 or num_examples`、`additional_forward_args must be a list`、`index out of range`、`input_shapes must be a non-empty list of shapes`、`modified_evals, masks and batch_sizes lengths must agree`、`initial_eval must be a number or an array`、`output should be a tensor whose 1st dim grow`、`modified eval cannot reshape to (-1, n_outputs)`、`single-input attribution needs exactly one input array`、`unknown forward_kind`、`additional_forward_args must be scalar numbers in attribution mode`、`Target list length does not match output!`、`multi forward needs an integer or list target`、`unknown mode`。

## 7. 判题用例设计

10 个 `check_*(solve)`：`check_mask_range`（默认/offset/3D/min>0/gap/trailing-1/大 ppe 钳制/bool quirk 共 8 组）、`check_mask_range_domain`（5 类坏 ppe + 6 类坏 mask + unknown mode）、`check_ablate`（6 组含张量 baseline、越界 start、行相关 baseline）、`check_ablate_domain`（7 条）、`check_schedule`（12 组含尾批重展开、全输入切片、extra_args tuple、ppe 钳制、min>0 进度）、`check_schedule_domain`（8 条）、`check_eval_out`（6 组含 int64 dtype、空记录、ppe>1 单记录、use_weights）、`check_eval_out_domain`（6 条）、`check_attribution`（20 组含六种 forward_kind、tuple、空输入、bias、min>0 进度）、`check_attribution_domain`（7 条）。driver 模拟 judge 注入 `solve`，参考解 **10/10 ALL ACCEPTED**。

## 8. mutation 覆盖

```text
257 raw mutants, 9 true equivalents removed, 248 active: 248 CAUGHT, 0 SURVIVED, 0 ERROR
ALL MUTANTS CAUGHT
```

首轮 SURVIVED 主要是删域约束后缺对应非法输入，以及若干「不可观测」点：`min>0` 的 `_mask_count`、`ppe` 大于特征数时的钳制、尾批张量附加参数按 K 展开、行相关张量 baseline、空输入跳过、标量前向带 bias 等。补精确消息断言与专门用例后全绿；另有重复锚点（`n_classes` 缩进、ppe 校验行）改成带上下文的唯一锚点。

9 个真等价 `EQUIVALENT`：`h-unflatten-scalar`（标量分支域内不可达）、`h-reshape-guard`、`h-bto-equal`、`ab-fullshape`、`ev-weights-float`、`at-shapevalid-once`、`at-usew-float`、`at-generate-div`、`at-generate-nodiv`——均由上游更强约束或同值变换兜住。

## 9. 常见踩坑

- 把 `ppe_clamped` 用 `max-min+1` 钳制，`min>0` 的 mask 会少算批次。
- `repeat_rows` 写成逐元素 repeat，尾批与满批的行序都会错。
- 尾批沿用 `ppe_eff` 份已展开输入 / 张量参数，或只切被消融的那一份输入。
- `diff` 忘记补 `[1]*(input_rank-1)` 广播维，3D 输入时形状直接崩。
- `broadcast_to` 对标量（`shape == []`）要单独处理，否则索引越界。
- `deltas` 式地把 `sum_axis0` 写成对全部扰动行求和，会把不同 chunk 的差值混在一起。
- 多输入默认 mask 未做全局 offset，`chunks` 的 start/end 会从 0 重启。
- `show_progress=False` 时返回了非 `None` 的 `progress_total`。

## 10. 自检清单

官方 12 段探针 87/87、三层对拍 5291 项 ALL GREEN、driver 10/10、mutation 257 原始 / 9 真等价 / 248 全 CAUGHT，随后才是四文档中英版、meta、starter 预检、11 文件落地、selftest 双模式、全库双语校验、collections 注册、todos 更新、短护栏与全量护栏。
