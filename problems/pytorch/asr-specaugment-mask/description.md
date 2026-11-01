# SpecAugment mask_along_axis 段生成与合并账

SpecAugment（Paper 1904.08779）对声学谱做 time/freq 两个轴的连续段置零
增强。FunASR 的 `funasr/models/specaug/mask_along_axis.py`（espnet 同源）
是 asr 训练管线里最常用的实现：它先用随机数采样每段的宽度和起点，再把
所有段的布尔行 `any` 到一起。本题把段的 `(起点, 宽度)` 显式入参，让你
逐条复刻段生成、布尔并合并与填充值的确定性部分。

## 接口

```python
solve(spec: list[list[float]], masks: list, axis: str = "time",
      replace_with_zero: bool = True) -> dict
```

- `spec`：`(T, F)` 声学谱；
- `masks`：段列表，每段 `[pos, width]`（官方随机采样的结果显式化）；
- `axis`：`"time"` 沿 T 轴（被 mask 的帧整行置值）或 `"freq"` 沿 F 轴
  （被 mask 的频点整列置值）；
- `replace_with_zero`：`True` 填 `0.0`；`False` 填**全谱所有元素**的
  算术平均（一个标量，在填充前对原始 spec 计算，含将要被 mask 的元素）。

返回：

```python
{"masked": list[list[float]],        # 置值后的谱（输入不改）
 "segments": [[start, end], ...]}    # 布尔并后连续区段，左闭右开、升序
```

## 规则

1. 轴维度 `D = T`（time）或 `F`（freq）；官方实现注释明说
   `spec_lengths` 未使用（不按有效长度截断）。
2. 段区间 `[pos, pos + width)` **左闭右开**：
   `mask[d] = (pos <= d) and (d < pos + width)`。`width == 0` 或整段
   越界（`pos >= D`）时该段为空；负起点截到 0、右端越界自动截断。
3. 多段通过**布尔并**合并：官方对每个候选段生成布尔行后 `any(dim=1)`。
   相邻段（`[0,2)` 与 `[2,4)`）与重叠段在布尔向量上自然连成同一区段，
   `segments` 就是这个布尔向量的连续段分解。
4. 填充值统一替换所有被 mask 的位置；`replace_with_zero=False` 的均值
   是全谱标量均值，不是逐行/逐列均值。

## 官方背景（理解用，不出题）

官方采样是：

```python
mask_length = randint(mask_width_range[0], mask_width_range[1])  # 上界开区间
mask_pos = randint(0, max(1, D - mask_length.max()))             # 全 batch 最大宽度耦合
```

即起点上限由**全 batch 最大段宽**决定（批内耦合），且宽度上界开区间
（`randint(0, 30)` 生成 0..29）。这些随机部分入参化后不在本题范围。

## 样例

```python
spec = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]  # T=4, F=3

solve(spec, [[1,2]], "time")
# 第 1、2 行整行置 0.0
# -> {"masked": [[1,2,3],[0,0,0],[0,0,0],[10,11,12]], "segments": [[1,3]]}

solve(spec, [[0,2],[2,2]], "time")
# 相邻段并成同一区段
# -> {"masked": [[0,0,0]]*4, "segments": [[0,4]]}

solve(spec, [[0,1]], "time", replace_with_zero=False)
# 全谱均值 = 6.5
# -> {"masked": [[6.5,6.5,6.5],[4,5,6],[7,8,9],[10,11,12]], "segments": [[0,1]]}
```

## 提示

- `segments` 是布尔并的结果：先并再找连续段，而不是对输入段做区间
  合并后罗列——但两者数学上等价，注意右开端点。
- 完全重复段、包含段对布尔并无影响；width=0 段与越界段一样是空段。
- freq 轴的 D=F：同一段列表在两个轴上语义不同（一个 mask 行、一个
  mask 列）。
