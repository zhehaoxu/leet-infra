# 871 · torchaudio flanger 延迟线效果账

本题重建 `torchaudio.functional.filtering` 的 flanger 效果链。官方逐字执行 `_generate_wave_table` 与 `flanger` 两个函数，判题侧用纯 Python 把 LFO 波表生成、延迟线环形缓冲、regen 反馈、线性/二次插值与 clamp 拆成八个可独立判题的 mode。全程确定性、无 RNG、无音频 IO，随机性与波表都只是显式公式。

## 回源与换题源

869 回源 `pytorch/audio main::src/torchaudio/functional/filtering.py`（1702 行 / 30 个顶层函数）并逐字执行 11 个 biquad 系数函数；870 从同一文件救回 `dither` 族。871 继续盘点该文件，取 `flanger`（713-868，156 行）与共用波表 `_generate_wave_table`（16-67，52 行）。

**回源纠错**：869 立项时把 `flanger` / `phaser` / `vad` 连同 `dither` 一起列入「RNG / 波表 / 逐样本循环」排除清单。实测 `flanger` 全程完全确定性、零 RNG：所谓「波表」只是 `torch.arange` + `sin` / 分段三角的闭式公式，所谓「循环」只是长度 T 的确定性状态机，二者都可纯 Python 逐位复现。

## 查重与划界

`flanger` / `_generate_wave_table` / `torch.frac` / `regen` / `lfo_length` 在全库 meta、description、notes 中只命中 869、870 两处的排除说明，不是覆盖。864 `au-torchaudio-speed-pitch-ledger` 判的是 `speed` / `pitch` / `phase_vocoder` 的重采样与相位编排；716 `asr-fsmn-vad-endpoint` 是 FunASR FSMN-VAD 端点状态机，与 SoX 效果器同名异义；869 判 biquad 系数、870 判 16-bit 量化与噪声整形账，871 均不重判。

## 官方逐字环境

`official871.py` 用 `ast.get_source_segment` 从官方 `.py` 抽两个函数的完整源码，`exec(compile(..., "filtering.py::<name>", "exec"), NS)` 进共享 namespace，官方函数体一字未改。注入 `torch`（`set_default_dtype(torch.float64)`）/ `math` / `typing` / `Tensor` / `F`。唯一插桩是 `_generate_wave_table` 的透明 spy：记录 `flanger` 实际传入的 keyword 参数与返回的整个波表张量，不替换任何数学。census 的默认值与参数名由 AST 的 `literal_eval` 独立读出。

## 机制与八个 mode

五路缩放：`feedback_gain=regen/100`、`delay_gain=width/100`、`channel_phase=phase/100`、`delay_min=delay/1000`、`delay_depth=depth/1000`；两级平衡 `in_gain=1/(1+delay_gain)`、`delay_gain=delay_gain/(1+delay_gain)*(1-|feedback_gain|)`。`delay_buf_length=int((delay_min+delay_depth)*sr+0.5)+2`、`table_min=floor(delay_min*sr+0.5)`、`table_max=delay_buf_length-2`、`lfo_length=int(sr/speed)`。

波表：`phase_offset=int(phase/pi/2*n+0.5)`、`point=(t+phase_offset)%n`；SINE 走 `(sin(point/n*2pi)+1)/2`，TRIANGLE 走 `point*2/n` 再按 `floor(4*point/n)` 的四段分派 `+0.5` / `1.5-d` / `1.5-d` / `d-1.5`；随后 `d*(max-min)+min`。`FLOAT` 分支强制 `.to(torch.float32)`，`INT` 分支保持 float64 并 `d±0.5` 后截断到 int32。

逐样本：`delay_buf_pos=(pos+L-1)%L` 递减，`cur_channel_phase=(channel_idxs*lfo_length*channel_phase+0.5).to(int64)` 截断，`torch.frac` 与 `torch.floor` 分离小数/整数延迟；写回 `temp+delay_last*feedback_gain`，取 `delayed_0/1(/2)` 三点，linear 走 `d0+(d1-d0)*frac`，quadratic 先把 `d1-=d0`、`d2-=d0` 再走 `a=d2*0.5-d1`、`b=d1*2-d2*0.5`、`d0+(a*frac+b)*frac`；输出 `waveform*in_gain+delayed*delay_gain` 后 `clamp(-1,1)` 并 `view(actual_shape)`。

八个 mode：`census`（源码普查、逐字默认值与参数名）、`wave_table`（波表逐值、dtype、phase_offset、points）、`flanger`（输出、shape、派生量、spy 记录的波表调用）、`branches`（两种调制 × 两种插值）、`params`（六参数各三值扫描）、`prefix`（因果前缀）、`errors`（七类异常类型与消息）、`shapes`（2D/3D/4D 的 view 打包）。

## 三个必须逐位复刻的 quirk

① **`torch.frac` 保留符号**：`torch.frac(x)=x-trunc(x)`，而 `torch.floor(x)` 向下取整。当 `delay<0` 使波表值跨负轴时二者不一致，整数延迟比小数延迟更负，插值实际是**向后外插**。用 `x-floor(x)` 复刻会在负 delay 配置上整片出错。

② **float32 波表降精度**：即便主链是 float64，`_generate_wave_table` 的 `FLOAT` 分支仍 `.to(torch.float32)`；`INT` 分支却保持 float64。参考解必须用 IEEE binary32 舍入逐位模拟（含负零与 nan），且 `INT` 分支不能套用该降精度。

③ **无 else 的双分支**：非法 `wave_type` 让 `d` 保持全零（波表恒为 `min`），非法 `data_type` 让 `d` 保持 float64 且不降精度——官方**不抛异常**。参考解若写 `raise` 会在域外行为上偏离，须显式复刻这两条静默落空分支。

## 判题口径

判题文件自包含：150 个官方常量快照 + 74 个 `check_*(solve)`，844 行，只用 `math` 与 `assert`，不导入 torch / numpy / 参考解。`_deep` 对 float 做 `repr` + 符号位比较，能区分 `-0.0`；非有限值一律归一成稳定字符串 `"nan"` / `"inf"` / `"-inf"`。判题域限制在 2D/3D/4D 波形、≤4 通道、`speed>0` 且 `lfo_length>0`；其余情形按官方异常类型与消息钉死。

## 流水线与证据

冒烟 `SMOKE 871 DONE`；12 段机制探针 **1159/1159 ALL GREEN**（独立闭式推导、Fraction 精确有理数、Newton 插值恒等式、deque 环形缓冲、前缀因果性）；两层对拍 **146138 checks / 2028 configs / 0 bad**；三层独立 oracle **1519 checks / 45 configs / 1429 props / 0 fails**（`cmath.exp` 正弦、Fraction binary32 半偶舍入、精确有理数缩放、Newton 三点插值）。

判题 `driver 74/74 ALL ACCEPTED`；`mut871.py` 194 项变异，两轮硬化捕获 11 个判题盲区（含 `params` mode 长波形高采样率、负 phase、四通道边界、非有限值清洗、非法波表类型、mode 级快照），剔除 9 个真等价后 **185 项有效变异全 CAUGHT / 0 SURVIVED / 0 ERROR**；`equiv871.py` 在 **1370 配置 × 9 mode × 9 mutant = 1271025 次**完整 `solve()` 输出精确比较（含异常、float 逐位、负零）证明零差异。

```text
census / wave_table / flanger / branches / params / prefix / errors / shapes
```
