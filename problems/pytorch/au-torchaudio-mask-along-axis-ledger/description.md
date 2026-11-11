# torchaudio SpecAugment 掩码账：`p==1.0` 直通与 `int` 截断、共享段的 reshape 打包与死代码宽度 guard、逐样本段的两次 transpose 广播与分离取整、Tensor `mask_value` 的 `where`/`repeat` 分支

## 1. 背景

`torchaudio.functional` 的 `_get_mask_param` / `mask_along_axis` / `mask_along_axis_iid` 是 SpecAugment 的频率掩码与时间掩码实现。它们只做参数解算、区间取整和掩码广播，不含任何谱计算或模型前向；随机性全部来自两次 `torch.rand`。本题把这三个官方函数用纯 Python 复刻成九个可独立判题的 mode，把 `torch.rand` 的抽样值改为显式入参，因此判题域完全没有 RNG。
全库查重：693 曾被记作覆盖 `mask_along_axis`，但回源确认 693 是 FunASR/espnet 的**显式段列表**账，与 torchaudio 这三个函数完全无关。855 覆盖谱/DB/deltas/CMN、857 相位声码器、859 sinc kernel、860 Griffin-Lim / ISTFT、861 loudness / biquad、862 MVDR、863 FID、864 变速/变调编排，函数集合均不重叠。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _get_mask_param        (line 806,  5 lines)
  mask_along_axis_iid    (line 813, 70 lines)
  mask_along_axis        (line 885, 74 lines)

  _get_mask_param:
    if p == 1.0: return mask_param
    else:        return min(mask_param, int(axis_length * p))

  shared / mask_along_axis:
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (size - value)
    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    if mask_end - mask_start >= mask_param: raise ValueError(...)

  iid / mask_along_axis_iid:
    value = torch.rand(shape[:dim-2], dtype=dtype) * mask_param
    mask_end = (min_value.long() + value.long())[..., None, None]
    specgrams = specgrams.transpose(axis, -1)
    torch.where(mask, mask_value.repeat(specgrams.shape), specgrams)
    specgrams = specgrams.transpose(axis, -1)
```

`official865.py` 用 `ast.get_source_segment` 抽出三个顶层函数并 `exec` 进同一共享 NS，官方函数体**一字未改**；唯一注入对象是转发型 `torch` 代理，仅把 `rand` 换成返回显式入参张量的记录器，其余属性（`arange` / `where` / `masked_fill` / `transpose` / `reshape` / dtype 等）全部走真实 torch，`Tensor` 仍是真 `torch.Tensor`，因此 `isinstance(mask_value, Tensor)` 分支保持活跃。

## 2. 口径约定

### 输入

谱是矩形数值嵌套列表（int / float 叶子合法，bool 与字符串拒绝，tuple 同样接受）；`variant` 为 `"iid"` 时走 `mask_along_axis_iid`，**其他任意字符串一律落到** `mask_along_axis`。`axis` 为 None 时解析为 `dim-1`（时间轴），只接受 `dim-2`（频率）与 `dim-1`（时间），负轴会被拒绝。`mask_param` / `mask_value` / `p` / `dtype` / `value_draws` / `start_draws` 全部显式入参，默认谱形状 `(2,3,4)`、默认 `mask_param=3`、`mask_value=0.0`、`p=1.0`、`dtype=float64`。
`value_draws` / `start_draws` 必须与 `torch.rand` 的记录形状逐元素对齐：shared 变体形状恒为 `[1]`，iid 变体形状为谱的前导维 `shape[:-2]`；数量不符返回逐字错误消息。shared 变体的 `torch.rand(1)` **没有 dtype 实参**，因此区间算术走全局默认 `float32`，即使谱是 `float64`；iid 变体传 `dtype=waveform.dtype`，所以跟随谱 dtype。纯 Python 参考解用 `struct.pack/unpack("f")` 逐标量步量化，逐位对齐 float32。

### 九个 mode

```text
mode            return value
--------------  ------------------------------------------------------
param           _get_mask_param(mask_param, p, sh[axis])
rand_calls      the two recorded torch.rand calls (shape/dtype/values)
interval        {value, min_value, start, end, width} per example
mask            half-open 0/1 row, nested over the leading dims
output          masked spectrogram, or the verbatim error dict
identity        {same_object, rand_calls, err}
errors          fixed invalid configurations -> verbatim messages
tensor_value    iid path with a real Tensor mask_value (repeat/where)
census          source-level facts about the three official bodies
```

`interval` / `mask` / `output` 在 `mask_param < 1` 的直通分支分别返回 `None` / `[]` / 原谱；`identity` 的 `same_object` 在该分支为 `True`，否则为 `False`。`output` 与 `tensor_value` 出错时返回 `{"type", "msg"}` 字典而不是抛异常；未知 mode 抛 `ValueError`，消息必须按 `MODES` 的原顺序列出全部九个名字。

## 3. 官方机制逐段

### 3.1 `p==1.0` 直通与 `int` 截断

`_get_mask_param` 在 `p == 1.0` 时**直接返回 `mask_param`**，完全不看轴长；否则返回 `min(mask_param, int(axis_length * p))`。注意 `int()` 是向零截断，而两个官方 docstring 都写 `floor(...)`：对 `(2,3,4)` 的谱，`p=0.5` 在时间轴（长 4）给 2、在频率轴（长 3）给 1；`p=0.999` 给 3，`p=0.39` 给 1。轴长必须取 `sh[axis]` 而非 `sh[-1]`，否则频率轴 `p<1` 会算错。
`p` 的合法区间是闭区间 `[0.0, 1.0]`，两端都接受；`p=0` 使 `mask_param` 变 0，命中 `mask_param < 1` 的直通分支返回**原对象**（不分配新张量、不消耗任何 rand draw）。`param` mode 直接调用该函数，因此可以用负 `p` 区分 `int(-1.2) == -1` 与 `floor(-1.2) == -2`。

### 3.2 shared 段的 reshape 打包与死代码宽度 guard

`mask_along_axis` 先把任意前导维 `reshape([-1] + shape[-2:])` 打包成 3D，再 `reshape(shape[:-2] + shape[-2:])` 解包回填，因此所有样本共享**同一个**掩码区间。掩码下标重映射为 `axis - dim + 3`；频率轴还要 `mask.unsqueeze(-1)` 才能广播到时间维。两个 `torch.rand(1)` 分别是 `value`（宽度）与 `min_value`（起点上界）的抽样。
区间端点为 `mask_start = min_value.long().squeeze()` 与 `mask_end = (min_value.long() + value.long()).squeeze()`：两次 `.long()` 是**分离取整**，`end = trunc(min_value) + trunc(value)`，而不是 `trunc(min_value + value)`。当两者小数部分之和超过 1 时会差 1（官方用 `value=0.27 * 10 = 2.7`、`start_draw=0.615384615...` 的配置可钉死 `end=2` 而合并取整给 3）。比较是半开区间 `start <= d < end`。
紧接着有 `if mask_end - mask_start >= mask_param: raise ValueError("Number of columns to be masked should be less than mask_param")`。因为 `value < mask_param` 且分离取整只会让宽度更小，这个 guard 在合法抽样下**永不触发**，只有显式注入 `value_draws > 1.0` 才能走到；它是官方的死代码，但错误消息必须逐字保留。iid 变体**没有**这个 guard，超宽注入掩码照样正常产出。

### 3.3 iid 段的两次 transpose 与 Tensor 分支

`mask_along_axis_iid` 要求至少 3 维，直接对前导维形状 `shape[:dim-2]` 各抽一次 `value` / `min_value`（`dtype=specgrams.dtype`），再用 `[..., None, None]` 扩出广播维。它先 `transpose(axis, -1)` 把待掩码轴换到最后，做完掩码后再 `transpose(axis, -1)` 换回；`mask` 由 `torch.arange(0, size(axis))` 生成。
当 `mask_value` 是真 `Tensor` 时走 `torch.where(cond, mask_value.repeat(specgrams.shape), specgrams)`——注意 `repeat` 用的是**转置之后**的形状；否则走 `masked_fill(cond, mask_value)`。标量张量与浮点值产出相同结果，但非标量张量会因 `repeat` 后的尺寸不匹配报错。`tensor_value` mode 强制使用 iid 函数并把 `mask_value` 包成真张量，用来钉死这条分支。

### 3.4 维度、轴与 p 的逐字校验

shared 要求 `dim >= 2`（消息含 `time and frequency`），iid 要求 `dim >= 3`，两者都报 `(%d given)` 的**维度数**而不是轴长。轴合法集合是 `[dim-2, dim-1]`，越界与负轴统一报 `Only Frequency and Time masking are supported (axis %d and axis %d supported; %d given).`；`p` 越界报 `The value of p must be between 0.0 and 1.0 (%s given).`。这些字符串全部来自官方逐字环境。

## 4. 代码骨架

```python
def solve(mode, **kw):
    c = _cfg(kw)                 # explicit draws, rectangular spec
    r = _run(c)                  # guards -> rand calls -> interval
    if mode == "param":     return r["param"]
    if mode == "rand_calls": return r["calls"]
    if mode == "interval":   return r["iv"]
    if mode == "mask":       return r["rows"]
    if mode == "output":     return r["err"] or r["out"]
    if mode == "identity":   return r["identity"]
    if mode == "tensor_value": ...   # force iid + real Tensor
    if mode == "errors":     ...     # fixed invalid configs
    if mode == "census":     ...     # source-level facts
    raise ValueError("unknown mode: ... (expected one of ...)")
```

参考解只用 `math` 与 `struct`：嵌套列表的矩形校验 / reshape 用手写 `_shape` / `_nest` / `_map1` / `_map2`，float32 用 `struct` 逐标量步量化，区间端点是 `math.trunc`（不是 `floor`），掩码应用是递归的 `_apply`。判题用例不含 torch、不 import 参考解、无全局 RNG，所有期望值都是官方环境 dump 出来的常量快照。

## 5. 关键 quirk 与易错点

1. **docstring 说 floor、代码是 `int()`**：`int` 向零截断，在合法非负区间与 floor 一致，但 `param` mode 的负 `p` 能区分两者。2. **shared 的 float32**：`torch.rand(1)` 不带 dtype，区间算术按全局默认 `float32`；iid 传 `dtype=specgrams.dtype`，所以 `dtype=float32` 的 iid 与 shared 在同一抽样下给出不同的 `min_value`。3. **分离 `.long()`**：`end` 是先各自取整再相加，不能合并成一次取整。
4. **死代码宽度 guard 只在 shared**：注入 `value_draws=1.5` 时 shared 报错、iid 正常。5. **`mask_param<1` 返回原对象**：不消耗任何 rand draw，`identity` 的 `same_object=True`。6. **Tensor `mask_value` 的 `repeat` 发生在转置之后**，且只有 iid 有该分支。7. **前导维不对称**：`(2,3,3,4)` 的形状下 `_map2` 的切片步长是 `prod(shape[1:])=3`，写成 `prod(shape[:-1])=2` 会错位；默认 draws 在 n≥5 时越过 1.0，`% 1.0` 的取模才可见。

## 6. 验证流水线

12 段官方机制探针 `380/380 ALL GREEN`（逐字源码比对、`p` 直通与截断、维度/轴/p 消息、rand 形状与 dtype、shared reshape 打包、分离取整、死代码 guard、iid 两次 transpose、Tensor repeat/where、负谱与单时间步、census、端到端账一致性）。三层对拍 `22054 checks / 74 configs / 0 fails`：官方走真实逐字环境，参考解走纯 Python 递归与 `struct` 量化，独立 oracle 走扁平索引算术 + 集合成员判定 + `Decimal ROUND_DOWN` 取整 + `ctypes.c_float` 量化，三条路径无共享算法。
判题 driver `140/140 ALL ACCEPTED`；169 项变异先补 13 条硬化用例（unknown-mode 消息的 MODES 顺序、identity 错误字典透传、空谱/空 draws 的 `[0]` 形状、字符串 spec 拒绝、`(2,3)` 前导维的取模与切片步长、负 `p` 的 trunc、频率轴 `p=0.5` 的区间、非零填充值、整数填充值的 float 类型）收敛到 3 个存活项，再用 `equiv865.py` 在 1901 组配置 × 10 个 mode 上逐项证明为真等价，最终 **169 项剔除 3 个文档化真等价 / 166 全 CAUGHT / 0 SURVIVED / 0 ERROR**。

## 7. 剥除口径与注册

判题域不含真实音频 IO、mel/STFT 数值、模型前向与训练；谱、`torch.rand` 抽样、`mask_value` 全部显式入参，纯 Python、无 numpy/torch/scipy、无全局 RNG。共享段与逐样本段的差异必须分别建账，尤其不能把 shared 的宽度 guard 复制到 iid，也不能把 `int()` 写成 `floor()`。注册后 `audio` 16→17、全库 842→843、order 池占用至 865，短护栏用独立 `--basetemp`，全量护栏后台运行后回填；按约定改完先不 git commit，等用户确认。
