# torchaudio ITU-R BS.1770-4 响度门控账：K-weighting 双 biquad 系数、clamp 只作用于返回值的 IIR 递推、0.4s/75% 重叠分块、绝对与相对双门限、5 声道权重

## 1. 背景

`torchaudio.functional.loudness` 按 ITU-R BS.1770-4 测量音频响度（LKFS）：先用两级 biquad 做 K-weighting，再按 0.4 秒窗、75% 重叠切块求能量，用绝对门限 -70 LKFS 与相对门限各筛一次，最后加权合成。本题把 `loudness` 以及它依赖的 `filtering.py` 里的 `treble_biquad` / `highpass_biquad` / `biquad` / `lfilter` 全链用纯 Python 重写，拆成五个可独立判题的 mode。
全库 `loudness` / `LKFS` / `BS.1770` / `biquad` / `lfilter` / `treble_biquad` / `highpass_biquad` 零命中；order 860 的 notes 里出现过 `loudness` 一词，但那只是一句「与 857 的 phase_vocoder 部分重叠」的候选避让说明，且与事实不符：响度链完全不调用 `phase_vocoder`，本题按官方源码订正该口径。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  loudness              (lines ~1535-1593)
pytorch/audio main::src/torchaudio/functional/filtering.py
  treble_biquad / highpass_biquad / biquad / lfilter
  _lfilter / DifferentiableFIR / DifferentiableIIR
  _lfilter_core_generic_loop
```

### 逐字执行环境

暂存文件 `official861.py` 用 `ast.get_source_segment` 从 855 已落盘的 2511 行 `functional.py` 抽取 `loudness`，再从回源落盘的 `filtering861.py` 抽取八个函数/类，全部 `exec` 到**同一个**命名空间，使 `loudness` 解析到真正的 `treble_biquad`，后者再解析到真正的 `biquad` → `lfilter`。注入 `math` / `warnings` / `torch` / `torch.nn.functional` / `Tensor` / `Optional`，并把 `_IS_TORCHAUDIO_EXT_AVAILABLE` 置为 False 使 `_lfilter_core_loop` 绑定到纯 PyTorch 的通用循环。任何官方函数体都未修改。
`comp_coeffs` 通过临时把命名空间里的 `lfilter` 换成记录器来捕获官方 biquad 生成的 `a` / `b` 系数三元组，函数体不改；`loudness_ledger` 是逐步复刻的**诊断镜像**（不是官方函数），用来读取中间账，`comp_loudness` 会断言镜像的 LKFS 与逐字 `loudness` 完全一致。

## 2. 口径约定

### 输入与输出

```text
solve(mode, **kw) -> dict
mode = coeffs | lfilter | kweight | blocks | loudness
```

参考解只暴露 `def solve(mode, **kw)`；判题文件顶层是 `def check_*(solve)`，由 judge 注入 `solve`，不得 `from solution import ...`。全部输入是 JSON 友好的数字与嵌套列表，纯 Python 实现，不依赖 `torch` / `torchaudio` / `numpy`，不使用全局 RNG。
★ `blocks` 与 `loudness` 的 `waveform` 恰为二维 `(channels, time)`；官方 `(..., channels, time)` 的批次维不纳入判题域，官方返回的 0 维 LKFS 张量在判题口径里是一个 Python float。`lfilter` 与 `kweight` 支持任意前导维并按原 shape 重建。`blocks` 接收**已经 K-weighted** 的波形，而 `loudness` 接收原始波形并自己跑两级 biquad。
判题数值全程 float64。官方系数与 `lfilter` 都跟随 `waveform.dtype`；float32 路径（系数被 `torch.as_tensor(..., dtype=float32)` 量化）只记录不判。独立 oracle 用 `math.pow` / `math.fsum` 与显式 padded 数组重写同一套公式，三层与官方误差约 1e-15。

### 非法输入

所有非法输入抛带具体关键字的 `ValueError`：`must be a non-empty list of numbers` / `must contain only numbers` / `must be a non-empty 2-D list` / `must have at least one column` / `must be rectangular` / `must be a non-empty list` / `must be an int` / `must be >=` / `must be a bool` / `a_coeffs must have exactly 3 numbers` / `b_coeffs must have exactly 3 numbers` / `a_coeffs[0] must be nonzero` / `Only up to 5 channels are supported.` / `waveform must have at least one channel` / `waveform shorter than gate_samples` / `step must be positive (sample_rate too small)` / `unknown mode`。校验顺序与精确消息都属于契约。

## 3. mode 1：coeffs（K-weighting 系数）

`treble_biquad` 是 SoX 风格的高频架式滤波器，`gain=4.0` dB、`central_freq=1500.0` Hz、`Q=1/sqrt(2)`；`highpass_biquad` 是 `cutoff_freq=38.0` Hz、`Q=0.5` 的二阶高通。两者都把 6 个系数交给 `biquad`，由它拼成 `a=[a0,a1,a2]` / `b=[b0,b1,b2]` 再调 `lfilter`。

```text
A     = exp(gain / 40 * ln(10)) = 10^(gain/40)
w0    = 2*pi*f0/sr            alpha = sin(w0)/(2*Q)
temp1 = 2*sqrt(A)*alpha      temp2 = (A-1)*cos(w0)
temp3 = (A+1)*cos(w0)
treble: b0=A*((A+1)+temp2+temp1)  b1=-2*A*((A-1)+temp3)
        b2=A*((A+1)+temp2-temp1)  a0=(A+1)-temp2+temp1
        a1=2*((A-1)-temp3)        a2=(A+1)-temp2-temp1
highpass: b0=b2=(1+cos(w0))/2  b1=-1-cos(w0)
          a0=1+alpha  a1=-2*cos(w0)  a2=1-alpha
```

返回 `treble` / `highpass` 各自的 `a` / `b` 原始系数、除以 `a0` 的 `a_norm` / `b_norm`，以及 `w0` / `alpha`；treble 额外暴露 `A` / `temp1` / `temp2` / `temp3`。高通满足 `b0==b2` 且 `a0+a2==2`。`sample_rate` 必须是 >= 1 的 int（bool 被拒）；官方不校验 Nyquist，故 sr=25 时 1500 Hz 远超奈奎斯特、系数仍有数值但滤波器不稳定，判题域只用 sr>=4000 跑波形。

## 4. mode 2：lfilter（IIR 差分方程）

官方 `lfilter` 先把 `b/a0` 送 `DifferentiableFIR`（`F.conv1d` 交叉相关，等价左补两个零的 FIR），再把 `a/a0` 送 `DifferentiableIIR`（逐样本反馈循环）。初始条件全零，递推为：

```text
z[n] = bn0*x[n] + bn1*x[n-1] + bn2*x[n-2]
y[n] = z[n] - an1*y[n-1] - an2*y[n-2]   (x[-1]=x[-2]=0, y[-1]=y[-2]=0)
out[n] = clamp(y[n], -1, 1) if clamp else y[n]
```

★ 关键 quirk：`clamp` 是在 `_lfilter` **返回之后**才由 `lfilter` 对整张输出张量做的，递推里保存的是**未 clamp** 的 `y[n]`。因此把 clamp 后的值反馈回去会改变后续样本，是常见错解。`lfilter` 还会把任意前导维 pack 成 `(batch, channels, time)`，算完再 reshape 回原 shape。

## 5. mode 3：kweight（两级 biquad）

`loudness` 依次执行 `treble_biquad(waveform, sr, 4.0, 1500.0, 1/sqrt(2))` 与 `highpass_biquad(treble, sr, 38.0, 0.5)`，两次都走 `lfilter(clamp=True)`。返回 `after_treble` 与 `after_highpass`（即 K-weighted 波形）以及所用系数。由于 clamp 夹在 [-1,1]，输入幅度超过 1 的波形会先被削顶再进入下一级。

## 6. mode 4：blocks（分块与双门限）

```text
gate_samples = int(round(0.4 * sr))   step = int(round(gate_samples * 0.25))
n_blocks     = (T - gate_samples)//step + 1
energy[c][b] = mean(x[c][b*step : b*step+gate_samples]^2)
g            = [1.0, 1.0, 1.0, 1.41, 1.41][:channels]
ew[b]        = sum_c g[c]*energy[c][b]
block_loudness[b] = -0.691 + 10*log10(ew[b])
gate_abs[b]  = block_loudness[b] > -70.0
efa[c]       = mean over gate_abs blocks of energy[c][b]
ew_abs       = sum_c g[c]*efa[c]
gamma_rel    = -0.691 + 10*log10(ew_abs) - 10
gate_rel[b]  = gate_abs[b] and block_loudness[b] > gamma_rel
efr[c]       = mean over gate_rel blocks of energy[c][b]
LKFS         = -0.691 + 10*log10(sum_c g[c]*efr[c])
```

★ `g` 的第 4、5 声道权重是 1.41（LFE），最多支持 5 声道；超过 5 直接抛 `Only up to 5 channels are supported.`。`unfold` 只取完整窗口，尾部不足 `gate_samples` 的样本被丢弃。绝对门限是**严格大于** -70；相对门限比绝对门限筛出的平均响度再低 10 dB。若某级没有任何块入选，`0/0` 得到 nan，随后 `log10(nan)` 仍是 nan，整条链路以 nan 收尾而不抛异常。
`step` 用 Python 的 `round`（银行家舍入）：sr=25 时 `gate_samples=10` → `round(2.5)=2`，sr=35 时 `gate_samples=14` → `round(3.5)=4`，直接截断会分别得到 2 和 3。全部块都通过两级门限时恒有 `LKFS == gamma_rel + 10`。

## 7. mode 5：loudness（完整链）

`loudness` = `kweight` → `blocks` 的全部门限账，额外返回 `after_treble` / `filtered` / `coeffs` / `shape`。官方逐字 `loudness` 对二维输入返回 0 维张量；判题口径把它剥成 float。示例（sr=4000，1 声道，2000 点方波）：`gate_samples=1600` / `step=400` / `n_blocks=2`，两块响度约 `[-6.794206, -6.783260]`，最终 `LKFS=-6.788729656623073`。

## 8. 官方 quirk 清单

① `waveform.size(-2) > 5` 检查声道数，一维波形直接 `IndexError` 而非 ValueError；② `treble_biquad` 用 `sin(w0)/2/Q`，`highpass_biquad` 用 `sin(w0)/2.0/Q`，写法不同但等价；③ `A = exp(gain/40*ln(10))` 而非 `10^(gain/40)` 的直写；④ `lfilter` 的 clamp 不参与递推；⑤ 两级 biquad 各 clamp 一次；⑥ `g` 张量的 dtype 跟随 waveform；⑦ `count_nonzero(gated_blocks, dim=-1)` 在 `(...,1,B)` 上只数块数；⑧ 绝对门限严格大于 -70；⑨ `gamma_rel` 用**绝对门限筛出的**能量算，不是全部块；⑩ 相对门限是 `logical_and` 而不是重新取阈值；⑪ 静音块 `log10(0)=-inf`，全被剔除时整链 nan；⑫ `gate_duration=0.4` / `overlap=0.75` / `gamma_abs=-70.0` / `kweight_bias=-0.691` 全为字面量常量；⑬ `sample_rate` 不校验 Nyquist。

## 9. 示例与落地

```text
solve('coeffs', sample_rate=8000)['highpass']['a']
  -> [1.0298406997386809, -1.9991093343177666, 0.9701593002613191]
solve('blocks', waveform=[[0.0]*10 + [1.0]*10,
                         [0.0]*10 + [0.5]*10], sample_rate=25)
  -> n_blocks=6, gate_abs=[False,True,True,True,True,True]
     LKFS=-1.9403873660829993
```

落地前确认 `max_order=860` / 838 题 / 861 空闲 / slug 未占用；`collections.json` 最小侵入注册，`todos.md` 更新为 839 题 / audio 13 / 下一批从 862 起。11 文件由 `land861.py` 落地，三件套与 `variants/pytorch/` 用 `filecmp` 校验一致。
