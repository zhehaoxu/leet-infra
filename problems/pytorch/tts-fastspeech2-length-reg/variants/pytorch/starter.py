# FastSpeech2 方差适配器（VarianceAdaptor）确定性核心账

FastSpeech2（2306.lei，官方实现 ming024/FastSpeech2 `model/modules.py`）在编码器输出上
先做三件确定性的事，再送解码器。本题复现这三件事的**逐算子口径**（不含任何神经网络前向，
predictor 输出直接作入参）：

## 输入

```
solve(log_durations, duration_target, d_control,
      expand_features, feat_value, p_control,
      feat_min, feat_max, n_bins, log_scale)
```

- `log_durations`: list[float] —— duration predictor 输出（log 域），长度 = phoneme 数 T
- `duration_target`: list[int] | None —— 训练模式给定的时长；给定时跳过预测链
- `d_control`: float —— 时长系数（推理侧加速/减速）
- `expand_features`: list[float] —— 每 phoneme 一个标量特征（代表编码器某维），
  展开后输出其序列
- `feat_value`: float —— pitch/energy 预测值（phoneme 级）
- `p_control`: float —— pitch/energy 系数，**乘在 bucketize 之前**
- `feat_min`, `feat_max`: float —— 统计文件中的 min/max
- `n_bins`: int —— 量化桶数（embedding 表大小）
- `log_scale`: bool —— True 时 bins 用 exp(linspace(log min, log max, n_bins-1))

## 输出 dict

```
{
  "duration_rounded": list,     # 每 phoneme 的最终整数时长
  "mel_len": int,               # 展开后总帧数
  "expand_indices": list,       # 展开后每帧来自的 phoneme 下标（长度 = mel_len）
  "feat_bucket": int,           # bucketize((feat_value, p_control) 乘积, bins)
}
```

## 官方口径（逐条对齐源码）

1. **时长链**（duration_target=None 时）：
   `duration_rounded = clamp(round(exp(logd) - 1) * d_control, min=0)`
   - 先 `exp(logd) - 1`（FastSpeech1 的 +1 偏置遗产）、torch.round（**银行家舍入**：
     round(2.5)=2、round(3.5)=4）、再乘 d_control、最后 clamp min=0。
   - d_control 非整数（如 1.5）时乘积可能非整数：LengthRegulator.expand 里
     `max(int(expand_size), 0)` 会**截断小数**（round(2)*1.5=3 整；round(2)*0.4=0.8 → int=0）。
   - duration_target 给定时 duration_rounded = duration_target 原样（官方分支）。
2. **展开**：逐 phoneme 重复其特征 `max(int(d), 0)` 次（0 时长不产生帧），
   拼接成 mel 序列；mel_len = 总帧数。expand_indices 记录每帧来源下标。
3. **桶索引**：`bucketize(value, bins)`，torch.bucketize 默认 right=False，
   等价 bisect_left：**value 恰等于 bins[k] 时落桶 k**（含下界）。
   bins = linspace(feat_min, feat_max, n_bins-1)；log_scale 时
   `exp(linspace(log(min), log(max), n_bins-1))`——**linspace 端点经 exp 后有浮点舍入**
   （如 min=50 时 bins[0]=49.999...<50，v=50 反而落桶 1），本题必须复现该浮点行为
   （bins 构造必须用 `min + i*(max-min)/(n-2)` 或等价公式后再 exp，禁用 Decimal）。
   bins 全程升序（min < max 保证）。

## 示例

```python
solve([math.log(3), math.log(1)], None, 1.0, [0.5, -0.5], 150.0, 1.0, 100.0, 300.0, 4, False)
# durations: round(e^log3-1)=2, round(e^0-1)=0 → [2, 0]
# expand_indices: [0, 0]（B 帧 0 个），mel_len=2
# feat: value*p_control=150 → bins=[100,200,300] → bucket 1
# -> {"duration_rounded":[2,0], "mel_len":2, "expand_indices":[0,0], "feat_bucket":1}
```

## 约束

- T = len(log_durations) = len(expand_features) ≥ 0；n_bins ≥ 2；feat_min < feat_max
- duration_target 给定时长度必须等于 T，元素为非负 int
- 所有中间值用 float64 语义；输出整数一律 int

写你的实现到 `solve`。
