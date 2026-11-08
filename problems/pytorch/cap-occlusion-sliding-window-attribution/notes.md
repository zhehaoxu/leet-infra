# Captum Occlusion 源码解剖

对应官方源码：

- `captum/attr/_core/occlusion.py::Occlusion.attribute`
- `captum/attr/_core/occlusion.py::Occlusion._occlusion_mask`
- `captum/attr/_core/feature_ablation.py::FeatureAblation.attribute` / `_process_ablated_out_full` / `format_result`
- `captum/attr/_utils/common.py::_format_and_verify_sliding_window_shapes` / `_format_and_verify_strides` / `_format_baseline`
- `captum/_utils/common.py::_select_targets` / `_verify_select_column`

## 一、官方 Occlusion 的完整流程

`Occlusion.attribute` 自己只做三件事：

1. 格式化并校验 `strides` 与 `sliding_window_shapes`；
2. 算出每一维的 `shift_counts`；
3. 把 `sliding_window_tensors`（每个窗口一个全 1 张量）和 `shift_counts` 交给父类 `FeatureAblation`。

真正的窗口枚举与归因在 `FeatureAblation` 的通用实现里。`Occlusion` 与 `FeatureAblation` 的**唯一关键差别**是：

```python
# FeatureAblation.__init__
self.use_weights = use_weights        # 默认 False

# Occlusion.__init__
super().__init__(model, multiply_by_inputs=multiply_by_inputs)
self.use_weights = True                # ← Occlusion 强制为 True
```

`use_weights=True` 就是「重叠区取平均」的来源。

## 二、★ 网格枚举：base-conversion，dim0 是最低位

官方 `_occlusion_mask`（`FeatureAblation` 里）：

```python
for current_index in itertools.product(*[
    range(shift_count) for shift_count in shift_counts
]):
```

这个 `itertools.product` 的语义等价于 base-conversion 计数：**最后一维变化最快**。但官方随后把它**转置**了：

```python
# 官方 _occlusion_mask 里的偏移计算
current_index = []
for i, shift_count in enumerate(shift_counts):
    current_index.append((remaining_total % shift_count) * stride)
    remaining_total = remaining_total // shift_count
```

这段是**手写的 base-conversion，dim0 是最低位**（变化最快）。实测确认：

```
shape=[3,4], window=[2,2], stride=[1,2] → 枚举顺序
(0,0) → (2,0) → (0,2) → (2,2)
```

`itertools.product` 给出的原始顺序是 `(0,0),(0,2),(2,0),(2,2)`（最后一维最快），官方额外做了转置，最终 dim0 最快。**这与直觉相反，是本题第一号陷阱。**

## 三、★ shift_counts 的公式与两条断言

```python
current_shape = np.subtract(inp.shape[1:], sliding_window_shapes[i])
assert (np.array(current_shape) >= 0).all(), "Sliding window dimensions ... cannot exceed input dimensions"
assert np.logical_or(
    np.array(current_shape) == 0,
    np.array(strides[i]) <= sliding_window_shapes[i],
).all(), "Stride dimension ... cannot be larger than sliding window shape dimension ..."
shift_counts.append(
    tuple(np.add(np.ceil(np.divide(current_shape, strides[i])).astype(int), 1))
)
```

- `shift_counts[d] = ceil((shape[d] - window_shape[d]) / stride[d]) + 1`。
- 断言一：窗口不得超过输入维。
- 断言二：**某维窗口未恰好覆盖该维时（`current_shape != 0`），stride 不得大于窗口长度**。
  ★ 例外：当 `shape[d] == window_shape[d]`（窗口恰好覆盖整维）时 `current_shape == 0`，stride 可以任意大 —— 此时 `shift_counts[d] = ceil(0/s)+1 = 1`，只枚举一个窗口。实测 `shape=[5], window=[5], strides=[7]` 合法。

## 四、★★ 窗口截断：F.pad 的负 padding

官方构造掩码的方式是把一个全 1 张量用 `F.pad` 定位到输入尺寸上：

```python
padding = []
for i in range(num_dims - 2, -1, -1):
    padding.append(current_index[i])
    padding.append(input_shape[i] - current_index[i] - sliding_window_shape[i])
mask = torch.nn.functional.pad(sliding_window_tensor, padding)
```

对第 `i` 维：左 padding = 偏移 `current_index[i]`，右 padding = `shape[i] - current_index[i] - window_shape[i]`。

★ **右 padding 可以是负数** —— `F.pad` 遇到负 padding 会**把窗口尾部裁掉**（等价于「从右侧裁掉 `-right_pad` 个元素」）。这正是末窗截断的机制。

实测 `shape=[6], window=[4], stride=[3]`：

- 窗口 0：左 0，右 `6-0-4 = 2` → `F.pad(ones(4), (0,2))` → 掩码 `[1,1,1,1,0,0]`
- 窗口 1：左 3，右 `6-3-4 = -1` → `F.pad(ones(4), (3,-1))` → **抛错？** 不，实测掩码为 `[0,0,0,1,1,1]`

`F.pad` 的负 padding 语义：负的右 padding 会**从张量末尾裁掉**元素。`ones(4)` 长度 4，左补 3 → 长度 7，右补 -1 → 裁掉末尾 1 个 → 长度 6。结果 `[0,0,0,1,1,1]` ✓ 与「覆盖区间 `[3, 6)`」完全一致。

⇒ 所以覆盖区间就是 `[start, min(start + w, shape[d]))`，与直接算区间等价。

## 五、★ `use_weights=True`：重叠区取平均

`FeatureAblation` 的核心累加（`_process_ablated_out_full`）：

```python
for feature in range(num_features):
    if ablation_masks[feature]:          # 该特征被本窗口覆盖
        attrib[feature] += attr_value
        weights[feature] += 1
```

然后 `format_result` 里：

```python
if use_weights:
    # single_attrib 是 float64 的累加值
    single_attrib = single_attrib.float() / weight
```

两处细节：

1. **`weights[feature]` 是覆盖该特征的窗口数** → 重叠区被多个窗口覆盖时，它们的差被**平均**。
   ★ 对照实验（1-D，`shape=[5]`, `window=[3]`, `stride=[1]`，模型是 `sum`）：
   - `Occlusion`（`use_weights=True`）：`[6.0, 7.5, 9.0, 10.5, 12.0]`
   - `FeatureAblation`（`use_weights=False`）：`[1.0, 2.0, 3.0, 4.0, 5.0]`
   两者相差**覆盖次数**的倍数。
2. ★ **`single_attrib.float()`**：官方先把 float64 的累加值**降级到 float32**，再在 float32 下做除法。所以归因结果不是精确的 float64 商。

第三处细节：**`weights` 全为 0 的位置**（没有任何窗口覆盖）会走 `0.0 / 0.0` → **NaN**。但在本题的口径下（窗口滑动 + stride ≤ window 保证全覆盖），`weights` 恒 ≥ 1，该分支不可达。

## 六、★★ 官方 `format_result` 的 float32 降级

```python
single_attrib = single_attrib.float() / weight
```

`single_attrib` 是 float64 的累加值，`.float()` 把它降级到 float32，然后除法也在 float32 下做。

这在归因值**不是二进制有限小数**时会产生可观测差异。例如累加值 `1.0`、权重 `3`：

- float64：`1.0 / 3 = 0.333333333333333314829616256247...`
- float32：`0.3333333432674407958984375`（float32 能表示的最接近 `1/3` 的值）

本题要求复现 float32 的结果。

## 七、★★ `target` 的负索引语义

官方 `_select_targets` 对 `int` 类型的 target 走：

```python
def _verify_select_column(output, target):
    target = (target,) if isinstance(target, int) else target
    assert len(target) <= len(output.shape) - 1, ...
    return output[(slice(None), *target)]
```

★ `output[(slice(None), -1)]` 就是 **PyTorch 的负索引** —— `target=-1` 选最后一个输出，`target=-2` 选倒数第二个。实测：

```
target=0  → [6.0, 7.5, 9.0, 10.5, 12.0]     （3 个输出模型的第 0 个）
target=1  → [12.0, 15.0, 18.0, 21.0, 24.0]
target=2  → [18.0, 22.5, 27.0, 31.5, 36.0]
target=-1 → [18.0, 22.5, 27.0, 31.5, 36.0]   ← 与 target=2 相同
target=-3 → [6.0, 7.5, 9.0, 10.5, 12.0]      ← 与 target=0 相同
target=3  → IndexError: index 3 is out of bounds for dimension 1 with size 3
target=-4 → IndexError: index -4 is out of bounds for dimension 1 with size 3
```

越界时官方抛 `IndexError`（PyTorch 的），本题统一用 `ValueError` 表达。

## 八、★ `strides` 的标量广播

官方 `_format_and_verify_strides`：

```python
if strides is None:
    strides = tuple(1 for input in inputs)
if len(inputs) == 1 and not (isinstance(strides, tuple) and len(strides) == 1):
    strides = (strides,)
```

★ 当只有一个输入张量、且 `strides` 不是「长度为 1 的 tuple」时，`strides` 会被包成 `(strides,)`。此时 `strides` 可以是一个**标量 int**，而 `sliding_window_shapes` 则是按维的 tuple —— 官方在 `_occlusion_mask` 里用 `strides[i]` 取值，标量情形下 `strides` 被 `_format_and_verify_strides` 之后的规范化逻辑转成按维的 tuple。

实测（`shape=[5]`, `window=[3]`）：

```
strides=1 -> [6.0, 7.5, 9.0, 10.5, 12.0]
strides=2 -> [6.0, 6.0, 9.0, 12.0, 12.0]     <- shift_counts = ceil(2/2)+1 = 2
```

2-D 情形（`shape=[3,4]`, `window=[2,2]`, `strides=2` 标量）：各维 stride 都是 2。

★ **1-D 下标量 stride 的「只改第一维」与「全部广播」等价**，所以判题必须用 2-D 用例才能盯住这条。

## 九、官方 `baseline` 的三种形态

`_format_baseline` 支持：

1. 与输入**同形**的 tensor；
2. 标量（广播）；
3. tuple of scalars（每个输入张量一个标量）。

本题口径：标量广播，或与展平后等长的列表。

★ 官方对 baseline 的长度校验是「broadcastable to match the dimensions of inputs」，本题简化为「长度必须等于 `prod(shape)`」。

## 十、判题设计的三层结构

1. **官方层**：直接调用 `captum.attr.Occlusion`，用 `float64` 网络（ReLU 只加在层间）→ 拿到权威输出。
2. **参考解层**：本题的纯 Python + `Fraction` 实现 → 拿到与官方**逐位精确相等**（`0.000e+00`）。
3. **独立 oracle 层**：不依赖参考解、只按题意重新实现一遍枚举与累加 → 交叉验证。

实测 14 个用例（1-D / 2-D / 3-D / 多输出 / 负 target / baseline 逐元素 / 截断）全部 **`0.000e+00`**。

## 十一、★ 本轮新经验：等价变异要主动剔除

mutation 跑到 46 项时，有一条存活：

```python
("target-negative-not-wrapped",
 "        return target + n_outputs",
 "        return target")
```

它把负索引折算 `target + n` 改成直接返回 `target`。**这条变异在语义上是等价的** —— 因为下游访问的是 Python 列表 `probe[tgt]`，而 Python 列表的负索引天然等价于 `n + target`。这是**真等价变异**（与 843 期遇到的 `xs-base-num-rejects-odd` 同类），应当删除而不是当成判题缺口去补用例。

替换成一条真正可观测的变异：

```python
("target-negative-clamped-to-zero",
 "        return target + n_outputs",
 "        return 0")
```

⇒ **mutation 存活时先问「这是语义缺口还是等价变异」**。等价的删除，缺口的补用例。

## 十二、★★ 本轮新经验：校验层要拆成互不依赖的「独立防线」

`x-length-check-dropped`（删掉「输入长度必须等于 shape 大小」的校验）最初**存活**了。原因：

- 该校验在 `_occlusion_core` 里（`len(flat_x) != total`）；
- 但 `_forward` 里还有一道「层输入维 vs 实际输入长度」的校验（`len(h) != n_in`）；
- 我的判题用例里，`x` 长度与 shape 不符的那条**恰好用了「层输入维 == shape 大小」的层**，于是两道校验指向同一个错误 → 删掉一道，另一道兜住 → 变异存活。

这正是 844 期总结的「独立防线」原则：同一语义约束不要在多个层次重复校验。

修法（把判题用例改成**只可能被目标校验抓住**的形状）：

```python
# 输入维为 2 的层
LAYER_2IN = [{"W": [[1.0, 1.0]], "b": [0.0]}]
...
_expect_error(
    lambda: solve("occlusion_diff", layers=LAYER_2IN, x=[1.0, 2.0], baseline=0.0,
                  shape=[5], window_shape=[3], strides=[1], target=0),
    "diff.x-length",
)
```

此时 `x` 长度（2）== 层输入维（2），但 ≠ `prod(shape)`（5）→ 只有「x 长度 vs shape 大小」这道校验会报错 → 删掉它必然被抓住（表现为 `IndexError` 而非 `ValueError`）。

⇒ **一道校验要能被 mutation 单独抓住，就必须在判题用例里制造一个「只有它会报错」的场景。**

## 十三、★★ 本轮新经验：1-D 抓不住的变异要在 2-D 补

`grid-scalar-stride-not-broadcast`（标量 stride 只广播到第一维）在 1-D 用例下**存活** —— 因为 1-D 只有一维，`[s] + [1] * 0` 与 `[s] * 1` 完全相同。

修法：补一条 2-D 标量 stride 用例

```python
rs2 = solve("occlusion_grid", shape=[3, 4], window_shape=[2, 2], strides=2)
# shift_counts = [2, 2]
```

⇒ **等价变异在低维下不可见**。凡是「广播 / 逐维」类语义，必须有 ≥2 维的用例。

## 十四、可复用的十条经验

1. **枚举顺序**：`itertools.product` 是最后一维最快，但官方的 base-conversion 是**第一维最快** —— 看代码，别猜。
2. **`F.pad` 的负 padding = 截断**，等价于把覆盖区间右端压到 `shape[d]`。
3. **`use_weights=True` = 重叠区取平均**，与 `FeatureAblation` 默认行为不同。
4. **float32 降级**：`.float()` 在除法**之前**，归因值不是精确的 float64 商。
5. **负 target 走 PyTorch 负索引**，`-1` 是最后一个输出。
6. **标量 strides 会广播到所有维**。
7. **`shift_counts` 的 `+1` 不能丢**。
8. **stride ≤ window 的例外**：窗口恰好覆盖整维时 stride 可以更大。
9. **校验必须是独立防线**：同一语义不要在多层重复校验，否则 mutation 存活。
10. **等价变异要剔除，维度不足的用例要补齐**。

## 十五、三层验证结构

```
official captum.attr.Occlusion  ->  reference solution (Fraction)  ->  independent oracle
        |                                     |                          |
   authoritative output            bit-for-bit identical        bit-for-bit identical
```

实测：**14 个用例全部 `0.000e+00`（逐位精确相等）**。
