# 857 解答笔记：torchaudio functional 波形算子账

## 1. 官方回源

### 源码清单与依赖

主源码是 `pytorch/audio main::src/torchaudio/functional/functional.py`（2511 行）。该文件在 order 855 已通过 GitHub API 回源并落盘到 `.workbuddy/scratch/captum855/functional.py`，857 直接复用，不重复拉取。抽取目标是九个顶层函数：`mu_law_encoding` / `mu_law_decoding`（约 668–735 行）、`_check_shape_compatible` / `_check_convolve_mode` / `_apply_convolve_mode` / `fftconvolve` / `convolve`（2189–2315 行）、`add_noise`（2317 行起）、`phase_vocoder`（732–802 行）。整库扫描确认 order 857 空闲、`au-torchaudio-waveform-ops-ledger` 无同名目录；`audio` 合集 8 题，与 855 的频谱后处理账按「波形侧 vs 频谱侧」划界，不重叠。

### 逐字执行环境

`official857.py` 用 `ast.get_source_segment` 按函数名整段抽取（含 docstring 与注释），`exec` 到共享命名空间 `NS`。`NS` 只注入 `math` / `typing` / `warnings` / `torch` 与 typing 别名，不 mock 官方函数体；驱动层把嵌套列表转 `float64` / `complex128` 张量：

```text
WANT = ["mu_law_encoding", "mu_law_decoding", "convolve", "fftconvolve",
        "_check_shape_compatible", "_check_convolve_mode",
        "_apply_convolve_mode", "add_noise", "phase_vocoder"]
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        exec(compile(ast.get_source_segment(source, node),
                     "functional.py::" + node.name, "exec"), NS)
```

驱动层提供 `comp_mu_law` / `comp_mu_law_decode` / `comp_convolve`（direct 与 fft 同源入口）/ `comp_apply_mode` / `comp_add_noise` / `comp_phase_vocoder` 六类对照入口。`comp_mu_law` 显式把 int64 编码值重新包成 `float64` 张量再解码，避开官方 `to(torch.float)` 的隐式 float32 降级；`comp_add_noise` 用与官方同式的 `vector_norm^2` 独立重算能量与 scale，便于把中间账暴露出来。

## 2. 十二段官方机制探针

### 探针结论

`probe857.py` 十二段共 78 项断言全绿：P1 mu-law 编码端点 / 中点 / int64 dtype / `+0.5` round-half-up / 四种 `qc`；P2 解码公式、端点、中点微小正数、单调性与 2D 形状；P3 直接卷积 full 数值、负核、`x` 更短时交换且数值不变；P4 `same` / `valid` 使用**原始**长度、交换后长度口径不变；P5 `_apply_convolve_mode` 的 valid / same / full 中心裁剪与非法 mode；P6 `fftconvolve` 与直接法在四组输入、三种 mode 下数值一致；P7 前导维成对广播、FFT 单例前导广播、rank 不符与 leading 不可广播两条域约束；P8 `add_noise` 标量 snr / 能量 / 0 dB 与 10 dB scale / `x + a*n` / snr rank 与长度不符；P9 逐行 snr 向量、`lengths` 只 mask 能量而 padding 区仍被加噪、`lengths` rank 与广播；P10 `rate=1` 直接返回输入（含嵌套对象与形状）；P11 `time_steps` / `alphas` / 帧数在四种 rate 下与 `arange(0,T,rate)` 一致、`rate=1.3` 的 ceil 规则；P12 相位累加、幅度插值、`phase_0` 保留、尾部零填充使幅度衰减、复数谱 2D 路径。

## 3. 三层对拍

### official 层

逐字 `exec` 的九个官方函数，覆盖 mu-law 四组 `qc`、卷积六组输入 × 三种 mode、FFT 四组、切片三种 mode、`add_noise` 标量/向量 snr 与 lengths、`phase_vocoder` 五种 rate 与批量前导维，统一 float64 / complex128。

### reference 层

`solution_staged.py` 六个 mode，纯 Python 无 torch / numpy；`shape_of` / `flatten` / `unflatten` / `broadcast_shapes` / `broadcast_to` / `slice_last` / `map_vals` 负责任意维度展平、广播与末维切片；`_dft` / `_idft` 是朴素 O(n^2) 复指数实现，用来复现 `rfft`/`irfft` 的乘积路径；`cmath.phase` / `cmath.polar` 对应官方的 `.angle()` 与 `torch.polar`。

### oracle 层

独立定义式：卷积用「逐输出位置三重循环」按定义累乘累加（不复用 `_conv_full`）；SNR 用 `sqrt(Es/En) * 10^(-snr/20)` 直接构造 scale；相位声码器用显式复数列表与逐帧 `cos`/`sin` 极坐标展开；mu-law 用独立的 `expm1`/`log1p` 等价式。三层在 961 项对照上完全一致。

### 对拍结论

```text
PROBE 857 DONE ok=78 fail=0 ALL GREEN
VERIFY 857 DONE checks=961 fails=0 ALL GREEN
DRIVER 857 12/12 ALL ACCEPTED   (before hardening)
DRIVER 857 20/20 ALL ACCEPTED   (after hardening)
```

## 4. 判题用例与补丁

### 首轮红灯

`test_cases_staged.py` 首轮 12 个 `check_*(solve)` 有 5 个 FAIL，逐条回源订正后确认**全是判题期望值写错，参考解与官方逐字环境完全一致**：

1. `check_mu_law`：把 float32 隐式降级路径的 `-0.4966767430305481` 当成期望，官方 float64 实为 `-0.4966766264665899`（对称位同步订正）；`x=0.3` 的 `0.30633397981616217` 经复核正确，保留。
2. `check_fftconvolve`：混合核卷积写成 `[0.15, 0.025, 0.275, 0.22, 0.08, 0.3]`，官方与手算均为 `[0.15, 0.025, 0.225, 0.325, -0.025, 0.3]`。
3. `check_phase_vocoder`：末组 `T=4, rate=1.3` 的帧数误写 3，官方 `arange(0,4,1.3)` 给 4 帧且末个时间点是 `3.9`。
4. `check_mu_law_domain`：把 `1 / 0 / -4` 放进「非整数」组，实际会先命中 `at least 2`；拆成非整数组 `(2.5, "8", True)` 与下界组 `(1, 0, -4)`。
5. `check_phase_vocoder_domain`：想测 rank 不足，却只把 `real` 改成 1D，先命中 `real and imag must have the same shape`；把 `imag` 同步改成 1D。

订正后 `driver857.py` 12/12 ALL ACCEPTED。

### 硬化补丁

首轮 mutation 暴露 42 个 SURVIVED，其中 18 项属于「行为已实现但判题观测不到」，`patch857.py` 追加 8 组不可观测用例并补两处断言：三维非对称前导广播（抓住 reversed / final 两次轴序反转）、单例前导维的末维不同取值（抓住广播索引与 build range）、等长操作数不交换且三抽头核（抓住 `b[i-k]` 负索引在双抽头核下碰巧等价、以及 `<=` 交换）、2D 正起点中心裁剪（抓住递归丢起点）、`y_length=1`、双非法输入的校验顺序、SNR 前导维冲突、bool 数组元素，另加 FFT 输出实数类型断言与 phase_vocoder 实部/虚部数值 + `freq` / `num_frames_in` 账。所有新增数值均先用官方逐字环境算出，再与参考解核对。补丁后 20/20 ALL ACCEPTED。

## 5. 变异测试

### 真等价清单

`mut857.py` 原始 201 项。25 项判为真等价并剔除，分四类：① 判题契约下不可达的分支（`h-shape-empty` / `h-reshape-guard` / `cv-size-zero` / `pv-empty` / `ml-dec-range` / `ml-dec-guard` 等，入口 `require_numbers` 已先拒绝空数组、解码只对合法编码值调用）；② 数学恒等式（`ml-log1p-inner` 的 `log1p` 与 `log(1+x)`、`ml-dec-expm1`、`sm-valid-abs`、`cv-direct` 的卷积可交换、`cv-fft-pada` / `cv-fft-padb` 的 DFT 隐式补零）；③ 当前调用形态下恒为同值（`rows-reshape` 的 `numel([])==1`、`cv-lead-one` 之后必被置空、`pv-rate1-ts` 的 `n==T`、`pv-pad` 一个零填充即够用、`h-bto-equal` / `h-bto-off` / `h-bto-rank` / `h-bto-scalar-value` / `h-unflatten-scalar` / `ml-dec-float` / `ml-x-guard` / `cv-size-msg`）；④ `pv-ph-wrap`：wrap 只给相位累加值加 `2*pi` 的整数倍，`cos` / `sin` 不变，另用 40 组确定性扫描（5 种 rate × 8 组 phase_advance）实测最大绝对差 `2.187e-14`，远小于判题容差。

### 最终结果

```text
MUTANTS: 201 total, 25 true equivalents, 176 active
176 CAUGHT, 0 SURVIVED, 0 ERROR  --  ALL MUTANTS CAUGHT
```

变异框架对 7 个在源码中出现两次的锚点加了 `ANCHOR_INDEX`（出现序号）以精确区分 `_slice_mode` 与 `_slice_mode_only`、以及两个 return dict 中的同名字段，避免替换到错误位置。

## 6. 官方 quirk 清单

1. `mu_law_encoding` 的 `.to(torch.int64)` 是向零截断；映射后非负，故 `+0.5` 实为 round-half-up。`mu_law_decoding` 对非浮点 `x_mu` 会 `to(torch.float)` 降到 float32，本题判题链路显式用 float64 re-wrap 规避。
2. `convolve` 在 `x` 尾维更短时交换操作数，但 `_apply_convolve_mode` 仍收原始 `x_size` / `y_size`；`fftconvolve` 不交换。二者用 `conv1d(weight.flip(-1))` / `rfft` 实现的是真卷积，不是互相关。
3. `_apply_convolve_mode` 的 `valid` 长度是 `max-min+1`，`same` 长度恒为 `x_length`；起点用整除，奇数 full 长偏左。
4. `add_noise` 的 `lengths` 只 mask 能量，输出用的仍是未 mask 的 `waveform` 与 `noise`，padding 区照样被加噪。
5. `phase_vocoder` 的 `rate == 1.0` 直接返回输入，跳过极坐标往返；`time_steps = arange(0, T, rate)` 故帧数是 `ceil(T/rate)`；时间补零长度是 2；相位先减 `phase_advance`、wrap、再加回，随后 `cat([phase_0, phase[..., :-1]])` 右移一帧并 `cumsum`。
6. 相位 wrap 到 `[-pi, pi]` 只改变 `2*pi` 的整数倍，对最终 `cos` / `sin` 数值等价（浮点差约 `1e-14`），这也是 mutation 里唯一需要数值实验才能判定的真等价。

## 7. 查重与复用

`audio` 合集 8→9 题；`au-torchaudio-spectral-postproc-ledger`（855）与本源同文件但只取频谱后处理链，本题只取波形侧九函数，两者 mode 集合、公式、判题字段无交集。复用 855 落盘的官方 `functional.py`，复用 856 的 precheck / land / register 脚本骨架与 `collections.json` 最小侵入注册口径。判题环境 `.venv-eval` 只有 torch + numpy，参考解与判题用例均不 import 任何第三方库，无全局 RNG。
