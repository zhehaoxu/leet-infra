# FastSpeech2 时长链与方差适配器账

## 题意

FastSpeech2（官方实现 `ming024/FastSpeech2`，`model/modules.py`）的 VarianceAdaptor
在编码器输出与解码器输入之间做三件确定性的事，predictor 的网络输出直接作入参，考察
输出之后的**逐算子账**：

1. **时长链**（`duration_target=None` 时）：
   `duration_rounded = clamp(round(exp(logd) - 1) * d_control, min=0)`
   - `exp(logd)-1`（FastSpeech1 遗产偏置）→ `round`（**银行家舍入**：2.5→2、3.5→4、0.5→0）
     → 乘 `d_control` → clamp min=0。四步顺序不可换。
   - `d_control` 非整数时乘积可非整数（2·0.4=0.8），官方 `LengthRegulator.expand`
     用 `max(int(size), 0)` **截断**——0.8 → 0 帧，加速语速会吞掉短音素。
2. **训练分支**：`duration_target` 给定时原样直通（不 round、不吃 d_control）。
3. **展开账**：逐 phoneme 重复其特征 `max(int(d), 0)` 次并拼接；0 时长不产帧；
   `expand_indices` 记录每帧来源下标，mel_len = 总帧数。
4. **桶索引**：`torch.bucketize` 默认 `right=False` = `bisect_left`：
   **value 恰等于 bins[k] 落桶 k**（100→0、200→1、300→2）。
5. **log bins 浮点端点 quirk**：bins = `exp(linspace(log min, log max, n_bins-1))`
   端点经 exp 有舍入（min=50 时 bins[0]=49.999…<50），**v=标称 min 的 50 反而落桶 1**；
   linear 模式端点精确，v=min 落桶 0。两种模式边界行为不同。
6. **p_control 乘在 bucketize 之前**：150×1.5=225 与 150×2.0=300（恰等上界→桶 2）
   的跨界效应要复现。

## 接口

```python
solve(log_durations, duration_target, d_control,
      expand_features, feat_value, p_control,
      feat_min, feat_max, n_bins, log_scale)
```

返回 dict：

```python
{
  "duration_rounded": list,   # 每 phoneme 最终时长（预测分支可为非整数浮点）
  "mel_len": int,             # 展开总帧数
  "expand_indices": list,     # 每帧来源 phoneme 下标
  "feat_bucket": int,         # bucketize(feat_value * p_control, bins)
}
```

约束：`len(log_durations) == len(expand_features)`；`duration_target` 给定时长度须等于
phoneme 数且元素非负 int；`n_bins ≥ 2`、`feat_min < feat_max`（n_bins=1 时 bins 为空，
bucket 恒 0）。

## 示例

```python
solve([math.log(3), math.log(1)], None, 1.0, [0.5, -0.5],
      150.0, 1.0, 100.0, 300.0, 4, False)
# durations: round(e^log3 - 1)=2、round(e^0 - 1)=0
# feat: 150 ∈ [100,200) → 桶 1
# -> {"duration_rounded":[2,0], "mel_len":2, "expand_indices":[0,0], "feat_bucket":1}

solve([math.log(3), math.log(4)], None, 0.4, [1.0, 1.0],
      0.0, 1.0, 1.0, 2.0, 2, False)
# round(2)*0.4=0.8 → int 截断 0 帧；round(3)*0.4=1.2 → 1 帧
# -> {"duration_rounded":[0.8,1.2], "mel_len":1, "expand_indices":[1], "feat_bucket":1}

solve([math.log(1)], None, 1.0, [1.0], 50.0, 1.0, 50.0, 800.0, 256, True)
# log bins 端点 49.99…<50 → v=50 落桶 1（浮点 quirk）
# -> {"duration_rounded":[0], "mel_len":0, "expand_indices":[], "feat_bucket":1}
```

## 数据规模与判题

- phoneme 数 T ∈ [0, 8]；n_bins ∈ [1, 256]；随机对拍 200 组（oracle 用 bisect_left
  与独立展开路径）。
- 变异覆盖：round→int 截断、去 -1 偏置、clamp 缺失、bisect_left→right、p_control
  乘→加、log bins 不 exp、linspace 端点偏移、target 分支吃 d_control 等 13 条全抓。
- 中间值按 float64 语义；`round` 必须是银行家舍入（Python 内建 / torch.round 语义一致）。
