# FastSpeech2 时长链与方差适配器账

## 背景与回源

FastSpeech2（2306.lei，官方实现 `ming024/FastSpeech2` 的 `model/modules.py`）是主流
TTS 声学模型。它在编码器输出与解码器输入之间插入 VarianceAdaptor，做三件确定性的事：
时长 → 帧数展开、pitch/energy 的量化桶嵌入。predictor 本身是卷积网络，但其**输出
之后的全部账目是纯标量/索引运算**，这正是本题的考察范围。注意官方 `LengthRegulator.expand`
是朴素逐 phoneme 重复（`vec.expand(max(int(size), 0))`），并无"余量按序分配"——
那是部分第三方实现的说法，本题按官方源码口径。

## 机制账

1. **时长链**（推理）：`duration_rounded = clamp(round(exp(logd) - 1) * d_control, min=0)`。
   四步顺序不可换：先 `exp(logd)-1`（FastSpeech1 的 +1 偏置遗产），再 torch.round
   （**银行家舍入**：round(2.5)=2、round(3.5)=4、round(0.5)=0），再乘 d_control，
   最后 clamp min=0。
2. **非整数时长截断**：d_control ≠ 1 时乘积可非整数（round(2)·0.4 = 0.8），官方
   `expand` 里 `max(int(size), 0)` 直接**截断**——0.8 → 0 帧，语速调快时小 phoneme
   可能整帧消失。
3. **训练分支**：`duration_target` 给定时 duration_rounded = target **原样**
   （不经过 round/控制链，d_control 不作用）。
4. **展开账**：逐 phoneme 重复特征 `max(int(d), 0)` 次后拼接；mel_len = 总帧数，
   0 时长 phoneme 不产生帧。
5. **桶索引**：`torch.bucketize(value, bins)` 默认 `right=False` = `bisect_left`：
   **value 恰等于 bins[k] 时落桶 k**（100→桶0、200→桶1、300→桶2）。
6. **log bins 浮点端点 quirk**：`exp(linspace(log min, log max, n_bins-1))` 的端点
   有舍入（min=50 时 bins[0]=49.999…<50），于是 **v = 标称 min 的 50 反而落桶 1**；
   linear 模式 linspace 端点精确，v=min 落桶 0。两种模式的边界行为不同。
7. **控制系数乘法位置**：p_control 乘在 bucketize **之前**
   （`prediction * control` 后再查桶）；乘 1.5 后恰等上界会跨界。

## 判题要点

- 12 条固定锚点 + 200 组随机对拍（oracle 用 bisect_left 与独立展开路径）。
- 变异 13 条全抓：round→int 截断、去 -1 偏置、clamp 缺失、bisect_left→right、
  p_control 乘→加、log bins 不 exp、linspace 端点偏移等；
  `max(int(d), -1)` 为数学等价变异（时长链已保证非负），不收录。
- linspace 必须**端点精确**（torch.linspace 语义），实现成 `lo + i*(hi-lo)/(num-1)`
  会与真 linspace 在中段浮点上有差——本题 oracle 用同一公式，对拍不受影响，
  但 log 端点 quirk 的锚点（v=50 落桶 1）依赖 `exp(linspace(log·))` 的实际舍入值。
