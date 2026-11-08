# 854 解答笔记：Captum Lime 可解释采样账

## 1. 官方回源

### 源码清单与依赖

主源码 `meta-pytorch/captum main::captum/attr/_core/lime.py`（1295 行 / 62667 bytes），依赖 `common.py`（927 行）、`attr_common.py`（386 行）、`batching.py`（245 行）、`attribution.py`（536 行）、`_utils/models/linear_model/model.py`、`_utils/models/model.py`、`progress.py`。`raw.githubusercontent.com/pytorch/...` 已全部返回 `HTTP=000`，改用 `api.github.com/repos/meta-pytorch/captum/contents/<path>?ref=main` 取 `content` 后 base64 解码。注意 `_utils/models/linear_model` 是**目录**而不是单文件。

### 逐字执行环境

```text
fake: captum._utils.typing / progress / log / attr._utils.attribution
      / _utils.models(.linear_model) / approximation_methods
exec: common.py -> attr_common.py -> batching.py -> lime.py   (shared NS)
inject: deterministic torch.bernoulli sequence
        explicit RidgeModel (fit / representation)
```

`log_usage` 用 `functools.wraps` 复刻并挂 `__wrapped__`，否则 `_attribute_kwargs` 里的 `super().attribute.__wrapped__` 会失效。`PerturbationAttribution` / `Attribution` 基类用最小替身（只保存 `forward_func` 与 `get_name`）；`SkLearnLasso` 是记录构造参数的 spy，用于验证默认 `alpha=0.01`。官方源码本身一行不改。

## 2. 十二段官方机制探针

### 探针结论

```text
P1  Lime.__init__: SkLearnLasso(alpha=0.01), cosine kw=1.0, Bernoulli 0.5
P2  Lime.attribute n_samples=25 (docstring says 50), ppe=1, return_input_shape=True
P3  construct_feature_mask: default per-scalar / global min shift / F = max+1
P4  default_perturb_func: bernoulli(ones(1,F)*0.5).long(), shape (1,F)
P5  default_from_interp_rep_transform: sample[0][feature_mask].bool()
P6  exp kernel: exp(-d^2/(2*kw^2)), cosine 1-cos / euclidean norm
P7  assembly: interp x / outputs / weights in sample order
P8  ppe scheduling: ceil(n/ppe) batches, loader batch_size = n_samples
P9  dataset tensors all float32, cat vs stack branches
P10 return_input_shape: coefs[f]*(mask==f) accumulation
P11 tuple inputs: cross-tensor same index shares coefficient
P12 bsz>1 per-example branch, no delta, multiplies_by_inputs False
```

最终 `PROBE 854 DONE: ok=76 fail=0`。

### float32 口径

P6 首轮 cosine 权重对不上，根因是官方把 `distance**2 / (2*kw**2)` 放在 **float32 张量**上算完再交给 `math.exp`，与纯 double 复算差约 `3.3e-9`；探针改为用同款张量运算复现期望后精确一致。P9 确认 `TensorDataset` 三列都是 float32；P7 的核权重按 `1e-6` 容差对拍。alpha>0 的用例里 `attribution` 还会因 `_convert_output_shape` 的 float32 累加差约 `2e-7`，三层对拍对该字段用 `1e-6`。

## 3. 三层对拍

### official 层

`official854.run_lime(cfg)` 注入确定性 Bernoulli 与可选显式 `weights`（替换 `similarity_func`），逐字跑 `Lime.attribute`，捕获 `RidgeModel.fit` 收到的 `(x, y, w)`、DataLoader 的 `batch_size`、原始 dtype 与最终 `out` / `coefs`。

### reference 层

`solution_staged.solve("lime", **cfg)`：纯 Python，Gram 与 rhs 用 `math.fsum` 聚合，线性方程组用 `Fraction` 高斯消元精确求解，全部中间量作为账本字段返回。

### oracle 层

`verify854.o_lime(cfg)` 与 `test_cases_staged._exp_lime(cfg)` 各自独立实现：不复用参考解任何辅助函数，从 cfg 直接规整 mask、构造扰动、算前向与核权重、用 `Fraction` 解方程、回填系数、算分批与形状元数据。

### 对拍结论

15 个用例覆盖 single/tuple、默认/平移/gap mask、linear/quadratic、alpha=0 与 alpha>0、自算 cosine/euclidean 核、显式权重、`return_input_shape` 开关、多种 `ppe` 与 `n_samples`。组件层还对拍 `construct_feature_mask` / `default_from_interp_rep_transform` / `get_exp_kernel_similarity_function`。最终 `检查项总数: 374 失败: 0 / RESULT: ALL GREEN`。

## 4. 官方 quirk 清单

### 默认值与 docstring quirk

`Lime.attribute` 的默认 `n_samples=25`，docstring 却写 50；`LimeBase.attribute` 的默认才是 50。默认 `SkLearnLasso(alpha=0.01)` 依赖 sklearn，`perturb_interpretable_space=True`、`to_interp_rep_transform=None`，`from_interp_rep_transform` 需要 `feature_mask` 与 `baselines` 两个 kwarg，缺任一即 assert。

### DataLoader batch_count

`batch_count += 1` 位于每个采样循环内且**从不清零**，于是 `DataLoader(dataset, batch_size=batch_count)` 的 batch_size 恒等于总样本数 `n_samples`，而不是 `perturbations_per_eval`。`_evaluate_batch` 仍按 `ppe` 分批调用，尾批单独补一次。`combined_outputs` / `combined_sim` 依据 `len(outputs[0].shape) > 0` 决定 `torch.cat` 还是 `torch.stack`。

### 回填 float32

`_convert_output_shape` 用 `torch.zeros_like(single_inp, dtype=torch.float)` 起底，再逐特征 `+= coefs[f].item() * (feature_mask == f).float()`；所有累加发生在 float32，与 double 系数解有约 `1e-7` 相对误差。tuple inputs 时对每个 tensor 各累加一遍，`_format_output` 决定是否返回 tuple。

## 5. 五个 mode 的实现要点

### feature_mask

`inputs` 规整为 tensor 列表；默认 mask 用累加 `idx` 逐标量编号，显式 mask 先校验非负 int 与形状，再按全局 `min` 平移，`F = max + 1`；返回 `mask / num_interp_features / shifted / default / min_index / shapes`。

### interp_transform

先拿规范化 mask，校验 `sample` 长度与 int 0/1，再用 `b = [sample[m] for m in mask_row]` 得到二值掩码、`p[j] = row[j] if b[j] else baseline[j]` 得到扰动输入；tuple 分支逐 tensor 记录 `binary` 与 `perturbed`。

### kernel

展平输入后按 `distance_mode` 算距离；cosine 用 `max(na*nb, 1e-8)` 兜底零向量，euclidean 用平方和开根；统一 `exp(-d^2/(2*kw^2))`；非法 mode / 宽度 / 长度都抛带关键字的 `ValueError`。

### surrogate

```text
X = [[float(v) for v in row] for row in samples]
gram = [[sum(W[i]*X[i][a]*X[i][b] for i in range(n))
         for b in range(F)] for a in range(F)]
for a in range(F): gram[a][a] += alpha
rhs  = [sum(W[i]*X[i][a]*Y[i] for i in range(n)) for a in range(F)]
coefs = solve_with_Fraction(gram, rhs)
```

奇异矩阵（例如两列完全相同且 `alpha=0`）抛 `surrogate Gram matrix is singular`；判题用例与对拍用例都避开退化设计。

### lime

串联 `feature_mask -> interp_transform -> _forward_value -> kernel/weights -> surrogate`，同时记录 `perturbed` / `binary_masks` / `batches` / `dataset_shapes` / `gram` / `rhs` 等 21 个字段；`return_input_shape` 决定回填还是直出 `coefs`。

## 6. 域约束与错误消息

`n_samples` / `perturbations_per_eval` 必须是正 int（bool 被拒）；`return_input_shape` 必须是 bool；`samples` 行数等于 `n_samples`、每行长度等于 `F`、元素为 int 0/1；显式 `weights` 长度与非负性；`distance_mode` 与 `kernel_width`；`forward_weights` 非空且覆盖每个特征；`alpha` 非负；`baselines` 形状；`feature_mask` 索引与形状；未知 mode 抛 `unknown mode`。判题用例对每条都断言**具体消息关键字**，避免被下游同款检查用相似消息兜住。

## 7. 判题用例设计

8 个 `check_*(solve)`：`feature_mask` / `interp_transform` / `kernel` / `surrogate` / `lime` / `domain` / `metadata` / `unknown_mode`。`lime` 用 13 个用例做全字段对拍，`metadata` 专门钉死 `batches` / `batch_count` / `loader_batch_size` / `dataset_shapes` / `default_mask` / `shifted` 等易被变异的元数据。所有显式序列逐元素变化，避免常数序列把布局变异伪装成真等价。

## 8. mutation 覆盖

158 项变异覆盖公共工具、`feature_mask`、`interp_transform`、`kernel`、`surrogate`、`_forward_value`、`lime` 的返回字段与 `solve` 分派。

```text
round 1: 130 CAUGHT / 24 SURVIVED / 4 ANCHOR-ERROR
fix    : 12 weaker domain assertions + 4 unique anchors + 2 true equivalents
final  : 156 CAUGHT / 0 SURVIVED / 0 ERROR  (ALL MUTANTS CAUGHT)
```

2 个真等价：`it-mt-count`（mask tensor 数已由 `feature_mask` 保证恒等于输入 tensor 数）与 `lm-samples-01`（`interp_transform` 有完全相同的 int 0/1 检查并以相同消息兜住）。

## 9. 常见踩坑

- 把 `feature_mask` 当布尔掩码、把 `F` 算成去重后的索引数。
- 指数核分母漏掉因子 2，或把 cosine 距离写成 `cos`。
- 用 `repeat` 布局重排样本，或让 `batch_count` 每批清零。
- 岭回归把 `alpha` 加到全矩阵、漏掉核权重、用非精确浮点解。
- 回填时按位置平铺系数而不是按 mask 分组；`return_input_shape=False` 时误返回输入形状。
- 判题用例里用常数权重/噪声序列，导致布局与切片变异不可观测。
- 域约束只断言宽泛关键字（如 `n_samples`），被下游 `samples length must equal n_samples` 之类的消息蒙混过关。

## 10. 自检清单

- 官方逐字环境与 12 段探针全绿（76/76）。
- 三层对拍 374 项 ALL GREEN；driver 8/8；mutation 156/156 CAUGHT。
- 中英 description / notes 的 heading 数、层级、fenced code block 数与内容一致；英文零 CJK。
- `meta.json` 十字段齐全且 `title_en` 零 CJK；starter / solution / test_cases 本地 `compile()` 通过。
- 落地 11 文件、selftest run/submit 双模式 ACCEPTED、全库双语零新增 warning、注册与 order 唯一、短护栏与全量护栏全绿。
