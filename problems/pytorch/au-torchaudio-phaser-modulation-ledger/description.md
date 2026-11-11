# 872 · torchaudio phaser 调制效果账

本题重建 `torchaudio.functional.filtering` 的 phaser 效果链。官方逐字执行 `_generate_wave_table` 与 `phaser` 两个函数，判题侧用纯 Python 把 INT 波表生成、双相位环形延迟线、decay 反馈、两级增益与 clamp 拆成八个可独立判题的 mode。全程确定性、无 RNG、无音频 IO，所谓波表与循环都只是显式公式与状态机。

## 回源与换题源

869 回源 `pytorch/audio main::src/torchaudio/functional/filtering.py`（1702 行 / 30 个顶层函数）并逐字执行 11 个 biquad 系数函数；870 从同一文件救回 `dither` 族；871 救回 `flanger`。872 继续盘点该文件，取 `phaser`（1206-1296，91 行）与共用波表 `_generate_wave_table`（16-67，52 行）。

**回源纠错（三连）**：869 立项时把 `phaser` 与 `flanger` / `vad` / `dither` 一起列入「RNG / 波表 / 逐样本循环」排除清单。870 实测 `dither` 主体完全确定性、871 实测 `flanger` 零 RNG，872 实测 `phaser` 同样全程确定性、零 RNG：INT 波表是`torch.arange` + `sin` / 分段三角的闭式公式，主循环只是长度 T 的环形 list 状态机，二者都可纯 Python 逐位复现，870 账中「循环 + 波表」的排除结论作废。

## 查重与划界

`phaser` 在全库只命中 869 / 870 / 871 的排除说明与注册账，`gain_in` / `mod_speed` / `mod_buf_len` 零覆盖。与 871 的关键划界：871 判 FLOAT 波表 + `frac` / `floor` 插值 + regen 两级增益平衡 + 最多 4 通道；872 判 INT 波表 + 整数索引无插值 + decay 单环反馈 + 两级增益 `gain_in` / `gain_out` + **无通道数限制**（实测 5 通道正常、一维输入也被 `view(-1, T)` 接受）。`phaser` 不调用 `biquad` / `lfilter` / `allpass_biquad`，故 869 的系数族、861 的 IIR 递推均不重判；864 speed/pitch、870 16-bit 量化、716 FSMN-VAD 亦不重判。

## 官方逐字环境

`official872.py` 用 `ast.get_source_segment` 抽两函数完整源码 `exec` 进共享 namespace，`torch.set_default_dtype(torch.float64)`，官方函数体一字未改。唯一插桩是 `_generate_wave_table` 的透明 spy：记录 `phaser` 实际传入的 `wave_type=SINE|TRIANGLE` / `data_type=INT` / `table_size=mod_buf_len` / `min=1.0` / `max=float(delay_buf_len)` / `phase=pi/2` 与整个 int32 波表，不替换任何数学。census 默认值与参数名由 AST `literal_eval` 独立读出。

## 八个判题 mode

`census` 钉源码结构与默认值；`wave_table` 判 SINE / TRIANGLE / 非法类型 × INT / FLOAT / 非法类型 × 相位 / 值域 / 尺寸；`phaser` 判完整输出与派生量；`branches` 判正弦/三角、decay 与 gain 分支；`params` 判六路参数扫描；`prefix` 判因果性；`errors` 钉七类边界异常；`shapes` 判 2D / 3D / 4D 打包。

## 核心机制

`waveform.view(-1, T)` 打包成 `(C,T)`；`delay_buf_len=int(delay_ms*0.001*sr+0.5)`、`mod_buf_len=int(sr/mod_speed+0.5)`；`sinusoidal` 决定波表类型。逐样本先读 `delay_buf_list[idx]` 再写 `delay_buf_list[delay_pos]=temp*decay`，`idx` 与 `delay_pos` 由两个独立相位推进；输出经 `stack` → `mul_(gain_out)` → `clamp(-1,1)` → `view(actual_shape)`。

```text
phase_offset = int(phase / pi / 2 * n + 0.5)
point       = (t + phase_offset) % n
SINE         d = (sin(point / n * 2 * pi) + 1) / 2
TRIANGLE     d = point * 2 / n ; floor(4 * point / n) -> 4 segments
scaled       = d * (max - min) + min
INT          v = scaled - 0.5 if scaled < 0 else scaled + 0.5 ; int32(v)
idx          = int((delay_pos + mod_buf[mod_pos]) % delay_buf_len)
mod_pos      = (mod_pos + 1) % mod_buf_len
delay_pos    = (delay_pos + 1) % delay_buf_len
temp         = waveform[:, i] * gain_in + delay_buf[idx]
out[:, i]    = clamp(temp * gain_out, -1, 1)
```

## 三个必须逐位复刻的 quirk

① **INT 分支向零截断**：`_generate_wave_table` 的 `FLOAT` 分支强制 float32，`INT` 分支却保持 float64 并先按 `d<0` 减 0.5、否则加 0.5，再 `.to(torch.int32)` **向零截断**（不是 `floor`）；负值 `min` 时两者会给出不同整数。② **读旧值再写**：`idx` 由 `delay_pos + mod_buf[mod_pos]` 得出，可能与写槽重合，官方先取旧值计算 `temp` 才替换 `delay_buf_list[delay_pos]`，顺序颠倒即错。③ **全零缓冲带符号**：官方先做 `delay_buf = delay_buf * decay`，`decay<0` 时初始全零缓冲每个单元都是 `-0.0`；`gain_in=0` 时输出的负零符号完全由它决定。

## 数值与异常边界

非法 `wave_type` / `data_type` 走官方「无 else」静默分支不抛异常（前者让 d 全零、后者保持 float64）。`mod_speed=0` → `ZeroDivisionError: float division by zero`；`sr=0` → `IndexError: list index out of range`；`delay_ms=0` → `RuntimeError: ZeroDivisionError`；`delay_ms<0` → `RuntimeError: zeros: Dimension size must be non-negative.`；`mod_speed<0` → `RuntimeError: upper bound and lower bound inconsistent with step sign`；ragged → `ValueError`。`clamp` 把 ±inf 饱和到 ±1，只有 nan 传播并归一为稳定字符串。

## 验证链

12 段机制探针 `15656 / 15656 ALL GREEN`；两层对拍 `8202 checks / 8202 configs / 0 bad`（类型敏感、float 逐位含 `-0.0`）；三层独立 oracle `63281 checks / 2151 configs / 44836 props / 0 fails`（SINE 走 `cmath.exp`、TRIANGLE 走 `Fraction`、binary32 走精确有理数、环形线用 dict 状态机独立重放）；**187 个官方常量 / 78 checks / 755 行自包含判题 / driver 78/78 ALL ACCEPTED**；**187 项变异剔除 4 个真等价后 183 项有效全 CAUGHT**（4 个由 1890 配置 × 9 mode = 516788 次完整 `solve()` 输出逐位比较证明零差异）。

## 工程教训

`int(x+0.5)` 类参数在整数值上不区分 `+0.5` 是否被篡改，判题必须挑跨舍入边界的小数（如 `delay_ms=3.45 / 3.499 / 3.7`、`mod_speed=277.0 / 500.0`）；短波形或 `mod_buf_len=2`（SINE 波表退化为常数）会让不同参数产生同一输出，须配长波形与高采样率扫描。torch 的向量化 float64 `sin` 与 libm 在少数 `n` 上差 1 ulp，判题常量避开这些尺寸、探针改用 1 ulp 容差断言；`float64 -> int32` 对 inf/nan 的结果依赖向量化尾路径，非有限值域只在 FLOAT / BYTE 分支验证。真等价变异不删除，而是对完整输出做类型敏感、float 逐位比较留证。
