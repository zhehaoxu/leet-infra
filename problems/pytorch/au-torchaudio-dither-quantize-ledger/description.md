# 870 · torchaudio dither 量化与噪声整形账

本题重建 `torchaudio.functional.filtering` 的 dither 系数量化链。官方逐字执行 `_apply_probability_distribution` / `_add_noise_shaping` / `dither` 三个函数，判题侧用纯 Python 把 16-bit 量化刻度、三种概率密度分支、半偶舍入与一阶误差反馈拆成八个可独立判题的 mode。无音频 IO、无隐藏随机源：`torch.randint` 的抽样值全部作为**显式入参**注入。

## 回源与换题源

855 回源的 `functional.py`（2511 行）里 55 个顶层函数已被 855–868 全部覆盖；869 改回源 `pytorch/audio main::src/torchaudio/functional/filtering.py`（1702 行 / 30 个顶层函数），逐字执行了 11 个系数函数。870 继续在同一文件里，取剩余最有实质内容的 `dither` 族三函数（行段 465-486 / 489-595 / 598-627）。

**回源纠错**：869 立项时把 `dither` 与 `flanger` / `phaser` 一起归入「RNG / 波表 / 循环」排除清单。实测 TPDF 主体完全确定性（`torch.bartlett_window` 三角窗 + 16-bit 量化），RNG 只出现在 RPDF / GPDF 分支，按 865 的 `torch.rand` 手法作 RNG spy 注入即可判；`_add_noise_shaping` 是纯确定性一阶递推。

## 查重与划界

`dither` 在题库的命中只有两处：852 `au-kaldi-fbank-compat-ledger` 的 Kaldi `_get_window` 加性噪声步骤（`noise[i][j]*dither`，同名异义，没有量化、TPDF 或噪声整形），以及 869 的排除说明。`bartlett_window` / `_apply_probability_distribution` / `_add_noise_shaping` / TPDF / RPDF / GPDF 在全库 meta、description、notes 零命中。

861 判 IIR 递推与 K-weighting，869 判 biquad 系数推导，857 判 `add_noise` 等波形运算，852 判 Kaldi 前端兼容链；870 **不重判上述任一**，只判 dither 的量化刻度、概率分布三分支、半偶舍入、RNG 调用序列与噪声整形递推。

## 八个 mode

| mode | content |
| --- | --- |
| `quantize` | 完整量化链：out / shape / rng_calls / bartlett / signal_scaled / density / noise_shaping |
| `bartlett` | 给定 n 的 periodic Bartlett 三角窗逐位值 |
| `round` | `torch.round` 半偶舍入与 `x*32766` 缩放 |
| `noise_shaping` | 独立的一阶误差反馈递推与形状账 |
| `branches` | TPDF / RPDF / GPDF 三分支输出与 RNG 调用序列 |
| `shapes` | 1D / 2D / 3D 输入的 reshape 打包与解包 |
| `errors` | 边界与异常（ragged / 空 / 非法密度 / 单通道 GPDF / T=1） |
| `census` | 源码结构常量表（30 个顶层函数、行段、四个刻度常量） |

## 核心机制

- 打包：`shape=waveform.size()`，`reshape(-1, shape[-1])` 成 `(C,T)`；`channel_size=C-1`、`time_size=T-1`。
- 刻度：`number_of_bits=16` → `up_scaling=2**15-2=32766`、`down_scaling=2**15=32768`；`signal_scaled = waveform * 32766`。
- `random_channel` / `random_time` 在分支**之前**先算（`channel_size>0` / `time_size>0` 才调 `torch.randint`，否则取 0），而 TPDF 分支算完却完全不用它们——典型的死代码 quirk。
- TPDF：`torch.bartlett_window(T, dtype=float64)` `repeat((C,1))` 后加到`signal_scaled`；RPDF：单点标量 `waveform[rc][rt]-0.5` 广播；GPDF：同一点起步，再累加 6 次 `waveform[rand_chan][rand_t]`。
- 三分支共走 `torch.round`（**半偶舍入**，且 `-0.0` 保留符号）→ `/32768` → `reshape(shape[:-1]+(T,))` 解包。非法 `density_function` 不报错，落到 `else` 的 TPDF 分支。
- `dither` 仅在 `noise_shaping=True` 时调 `_add_noise_shaping`：`error[n]=dithered[n]-waveform[n]`，`error_offset=cat((0, error))` 截到 T，`out[n]=dithered[n]+error[n-1]`，首样本误差丢弃。

### 两个必须逐位复刻的浮点 / 别名 quirk

**Bartlett 分段浮点**：periodic 窗长度 N=n+1、因子 `f=2/n`，左半 `w[i]=i*f`（`i<=n//2`），右半 `w[i]=2-i*f`。两半用不同浮点表达式，故奇数 n 不是逐位对称：`n=5` 给 `[0.0, 0.4, 0.8, 0.7999999999999998, 0.3999999999999999]`；`n=98` / `196` 时 `n//2` 与 `(n-1)//2` 还差 1 ulp。`n=1` 特例 `[1.0]`。

**GPDF in-place 视图别名**：`gaussian = waveform[rc][rt]` 是 0 维**视图**，`gaussian += ...` 原地改写输入波形。当 6 次抽样都落在同一点时该点每轮翻倍，`0.1 -> 6.4`；更隐蔽的是，`dither(noise_shaping=True)` 随后把这个**已被改写**的波形当作 original 传给 `_add_noise_shaping`，误差反馈因此出现大负值。

## IEEE 边界与判题域

`inf` / `-inf` / `nan` 不是合法 Python 字面量，fixtures 与参考解一律把非有限值归一为稳定字符串；ragged 波形、空张量、嵌套空、单通道 GPDF、`T=1` 的错误类型与消息都按官方逐字钉死。判题域约定 `dithered` 与 `waveform` 同形（它本就是同一次`dither` 的输出）。

## 流水线留痕

| stage | result |
| --- | --- |
| 官方逐字环境 | `ast.get_source_segment` 抽 3 函数 exec，`torch(float64)` + RNG spy，冒烟 SMOKE 870 DONE |
| 12 段探针 | 461 / 461 ALL GREEN |
| 两层对拍 | 32035 checks / 822 configs / 0 bad |
| 三层独立 oracle | 6466 checks / 441 configs / 3050 props / 0 fails |
| 判题 | 110 官方常量 / 59 checks / 618 行 / driver 59/59 ALL ACCEPTED |
| 变异 | 171 项，剔除 11 个真等价后 160 项全 CAUGHT / 0 SURVIVED / 0 ERROR |
| 等价证明 | 1214 配置 x 9 mode x 11 mutant = 617507 次精确比较零差异 |

## 已排除

`filtfilt`（只两次 `lfilter` + flip，本体过薄）、`flanger` / `phaser` / `overdrive` / `_generate_wave_table`（逐样本循环 + 波表）、`gain` / `contrast` / `dcshift`（过薄）、`vad` / `_measure`（与 `asr-fsmn-vad-endpoint` 重叠）。
