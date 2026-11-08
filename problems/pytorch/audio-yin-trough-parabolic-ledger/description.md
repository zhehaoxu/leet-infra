# librosa YIN 的 CMND 累积均值归一化差分、抛物线插值与谷值阈值账

## 背景

YIN（de Cheveigné & Kawahara, 2002）用「累积均值归一化差分函数」（CMND）替代自相关来估计基频。librosa 的 `yin()` 把它拆成三段可独立验证的数值账：

1. `_cumulative_mean_normalized_difference`：把自相关与平方能量变成归一化差分 `d'(tau)`；
2. `_parabolic_interpolation`：在离散谷值附近做分段抛物线插值，得到亚 bin 精度的偏移；
3. `util.localmin` + 绝对阈值：挑出第一个低于阈值的谷值，回退到全局最小，再换算成周期与频率。

本题只重建这三段**纯数值**账。FFT 自相关、`pyin` 的 HMM/Viterbi 概率解码、`piptrack` 与真实波形 IO 都不在判题域内——输入一律是已经算好的数值列表。

## 通用约定

- 所有输入都是 Python `float` 组成的列表；不涉及张量、梯度和批处理。
- 浮点比较采用绝对误差 `1e-9` 加相对误差 `1e-9`。
- `tiny` 指 `numpy.float64` 的最小正规数 `2.2250738585072014e-308`，即官方 `util.tiny()` 的返回值。
- 返回值统一是 `dict`，字段名见各 mode 说明。
- 下标一律从 0 开始，且**闭区间**。

## 模式与规则

### 1. `cmnd`：累积均值归一化差分

入参：

| 参数 | 含义 |
|---|---|
| `sq` | 一帧的平方信号 `y_m^2`，长度不少于 `max_period + 1` |
| `acf` | 自相关 `r[0..max_period]`，长度恰好 `max_period + 1` |
| `min_period` | 最小周期，`1 <= min_period <= max_period` |
| `max_period` | 最大周期 |
| `tiny` | 分母保护项，默认 `2.2250738585072014e-308` |

计算步骤：

```
E[k]     = cumsum(sq)[k]            # 平方能量前缀和
E[0]     = 0                        # 官方 quirk：写 d 之前先把 E[0] 抹成 0
d[0]     = 0
d[k]     = 2 * (acf[0] - acf[k]) - E[k - 1]        # k = 1 .. max_period
cummean[j] = (d[1] + ... + d[j+1]) / (j+1)       # j = 0 .. max_period-1
den[i]     = cummean[min_period - 1 + i]          # i = 0 .. max_period-min_period
cmnd[i]    = d[min_period + i] / (den[i] + tiny)
```

返回字段：

- `difference`：长度 `max_period + 1`，即 `d[0..max_period]`；
- `cumulative_mean`：长度 `max_period`，即 `cummean[0..max_period-1]`；
- `denominator`：长度 `max_period - min_period + 1`，即 `den`；
- `cmnd`：长度 `max_period - min_period + 1`，即归一化差分本身。

易错点：

- `E[0]` 必须先置 0 再算 `d`，否则 `d[1]` 会多减 `sq[0]`；
- `d[k]` 减的是 `E[k-1]`，不是 `E[k]`；
- `cummean` 的分母是 `k_range = [1, 2, ..., max_period]`，不是 `tau`；
- `den` 的切片起点是 `min_period - 1`；
- `tiny` 加在**分母**上，不是分子。

### 2. `parabolic_shifts`：分段抛物线插值

入参 `x`：CMND 值序列，长度 `n >= 3`。

对每个内部下标 `i = 1 .. n-2`：

```
a[i]     = x[i+1] + x[i-1] - 2 * x[i]
b[i]     = (x[i+1] - x[i-1]) / 2
shift[i] = 0            若 |b[i]| >= |a[i]|
         = -b[i] / a[i]  否则
```

首尾 bin 强制为 0：`shift[0] = shift[n-1] = 0`。

返回字段：`a`（长度 `n`）、`b`（长度 `n`）、`shifts`（长度 `n`）。`a` 与 `b` 的首尾元素为占位 `0.0`。

易错点：

- 抑制条件是 `|b| >= |a|`（含相等），此时偏移取 0；写成 `>` 会在 `|b| == |a|` 时给出错误的非零偏移；
- `a = 0` 时必然落入抑制分支（`|b| >= 0` 恒真），不会出现除零；
- 首尾 bin 是显式置 0，不是插值结果。

### 3. `trough_selection`：谷值挑选与绝对阈值

入参：

| 参数 | 含义 |
|---|---|
| `x` | CMND 值序列，长度 `n >= 3` |
| `min_period` | 最小周期 |
| `sr` | 采样率 |
| `trough_threshold` | 绝对阈值 |

计算步骤：

```
is_trough[i] = (x[i] < x[i-1]) and (x[i] <= x[i+1])   # i = 1 .. n-2
is_trough[n-1] = x[n-1] < x[n-2]                      # 末元素没有右邻居
is_trough[0]   = x[0] < x[1]                          # 官方覆盖第 0 个 bin
is_threshold_trough[i] = is_trough[i] and (x[i] < trough_threshold)
global_min = argmin(x)                                # 并列取第一个
yin_period = argmax(is_threshold_trough)              # 第一个 True
if 没有任何一个 True: yin_period = global_min
shift  = shifts[yin_period]
period = min_period + yin_period + shift
f0     = sr / period
```

返回字段：`is_trough`、`is_threshold_trough`（均为长度 `n` 的布尔列表）、`global_min`、`yin_period`、`no_trough_below_threshold`（布尔）、`shift`、`period`、`f0`。

易错点：

- `util.localmin` 的条件是**左严格 `<`、右非严格 `<=`**，两侧不能互换；
- 末元素用 `x[n-1] < x[n-2]`，两侧都严格；
- 第 0 个 bin 由官方显式覆盖为 `x[0] < x[1]`；
- 阈值比较是 `x[i] < trough_threshold`，**等于阈值不算**；
- `argmax` 取**第一个** True，`argmin` 并列也取第一个；
- 全部为 False 时必须回退到 `global_min`；
- `period = min_period + yin_period + shift`，`f0 = sr / period`。

## 输入与返回示例

`cmnd`：

```python
solve(
    "cmnd",
    sq=[1.0, 4.0, 9.0, 16.0, 25.0],
    acf=[55.0, 34.0, 20.0, 10.0, 4.0],
    min_period=2,
    max_period=4,
)
# {
#   "difference":      [0.0, 42.0, 65.0, 76.0, 72.0],
#   "cumulative_mean": [42.0, 53.5, 61.0, 63.75],
#   "denominator":     [53.5, 61.0, 63.75],
#   "cmnd":            [65.0/53.5, 76.0/61.0, 72.0/63.75],
# }
```

`parabolic_shifts`：

```python
solve("parabolic_shifts", x=[1.0, 0.2, 0.5])
# {
#   "a":      [0.0, 1.1, 0.0],
#   "b":      [0.0, -0.25, 0.0],
#   "shifts": [0.0, 0.25/1.1, 0.0],
# }
```

`trough_selection`：

```python
solve(
    "trough_selection",
    x=[0.5, 0.05, 0.3, 0.02, 0.4],
    min_period=2,
    sr=100.0,
    trough_threshold=0.1,
)
# {
#   "is_trough":                [False, True, False, True, False],
#   "is_threshold_trough":      [False, True, False, True, False],
#   "global_min":               3,
#   "yin_period":               1,
#   "no_trough_below_threshold": False,
#   "shift":                    0.1/0.7,
#   "period":                   3.0 + 0.1/0.7,
#   "f0":                       100.0/(3.0 + 0.1/0.7),
# }
```

## 不进判题域

- `librosa.util.localmin` 之外的 `peak_pick` / `localmax` / 贪心峰值挑选；
- `audio.autocorrelate` 的 FFT 实现与多帧广播；
- `pyin` 的 HMM/Viterbi 概率解码、随机采样与音高平滑；
- `piptrack` 的谱峰估计与 `__check_yin_params` 的异常分支；
- `center` / `pad_mode` 分帧、多通道处理；
- 真实波形文件读取与重采样。
