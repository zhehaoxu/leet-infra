# 官方机制与易错点笔记

## 官方来源与版本

- 仓库：`librosa/librosa`，默认分支 `main`。
- 用到的文件：`librosa/core/pitch.py`（`_cumulative_mean_normalized_difference`、`_parabolic_interpolation`、`_pi_stencil`、`yin`）与 `librosa/util/utils.py`（`localmin`、`_localmin_sten`、`tiny`）。
- 回源方式：GitHub API `git/trees/main?recursive=1` + `contents?ref=main` + `base64.b64decode`。
- 判题环境（`.venv-eval`）里**没有** librosa 与 numba，因此官方对拍脚本把 numba stencil 逐行展开成等价的纯 numpy/Python，公式与源码严格对齐；`tiny` 取 `np.finfo(np.float64).tiny`。

## CMND 段的四个坑

1. **`E[0]` 置零 quirk**：官方先对平方信号做 `cumsum`，然后立刻 `yin_frames[..., 0, :] = 0`，再写 `d[k]`。所以 `d[1] = 2*(r0 - r1) - 0`。若不抹零，`d[1]` 会多减 `sq[0]`，后续所有 `cummean` 全部偏移。
2. **`- E[k-1]` 的下标**：差分减的是**能量前缀和**而不是差分自身，且下标是 `k-1`。numpy 先求右侧临时数组再写回，所以 `d[k]` 用的仍是原始 cumsum 值。
3. **`cummean` 的分母是 `k_range = [1, 2, ..., max_period]`**，即差分个数，不是 `tau`。写成 `tau` 会在 `min_period != 1` 时整体错。
4. **`den` 的切片偏移是 `min_period - 1`**：`den = cumulative_mean[min_period-1 : max_period]`，长度 `max_period - min_period + 1`。这是最容易少写一个 `-1` 的地方。

另外 `tiny` 加在**分母**上（`yin_numerator / (yin_denominator + tiny)`）。正常输入下它对数值几乎无影响，但 `den` 全零时没有它就会直接除零，所以这一项也要判。

## 抛物线插值段的两个坑

1. **`|b| >= |a|` 是含等价的抑制**。官方 stencil 写的是 `np.abs(b) >= np.abs(a)`，相等时直接返回 0。改成严格 `>` 会在 `|b| == |a|`（例如 `x = [-0.5, 0.0, 1.5]`，此时 `a = b = 1.0`）时给出错误的非零偏移。`a = 0` 时因为 `|b| >= 0` 恒真，落入抑制分支，不会除零。
2. **首尾 bin 是显式置 0**。`_parabolic_interpolation` 在 stencil 之后执行 `shiftsi[..., -1] = 0` 与 `shiftsi[..., 0] = 0`。numba stencil 的默认边界模式本身就输出 0，所以这两行是冗余但显式的；判题时仍然按「首尾恒为 0」处理，并且要让 `yin_period` 落在首/尾 bin 的用例参与校验（此时 `shift` 必须是 0，`period` 才等于 `min_period + idx`）。

## 谷值挑选段的四个坑

1. **左严格、右非严格**：`util.localmin` 的条件是 `x[i] < x[i-1]` 且 `x[i] <= x[i+1]`。这与 `localmax` 相反，两侧不能互换。判题网格里必须**混入并列值**（把随机值量化到固定刻度），否则 `<` 与 `<=` 的差别在纯随机浮点里几乎测不出来。
2. **末元素特判**：没有右邻居，官方用 `xi[..., -1] < xi[..., -2]`，两侧都严格。
3. **第 0 个 bin 被官方覆盖**为 `x[0] < x[1]`（stencil 的边界原本恒为 False）。
4. **阈值是严格小于**：`x < trough_threshold`，等于阈值不算谷值。`argmax` 取**第一个** True，`argmin` 并列也取第一个；全部为 False 时 `argmax` 会返回 0，但必须被 `global_min` 覆盖。

最后 `period = min_period + yin_period + shift`，`f0 = sr / period`。注意 `min_period` 是 tau 轴起点，不能漏。

## 验证方式与判题设计

- 官方回源对拍：三个 mode 各 600 组，共 **1800 组全一致**。
- 判题用例：每个 mode 一个 `check_*`，内含独立纯 Python oracle、若干手算锚点与网格对拍。
- 手算锚点专门钉住这些细节：`E[0]` quirk、`-E[k-1]` 下标、`cummean` 的 `k_range` 分母、`den` 的 `min_period-1` 偏移、`tiny` 加在分母、`|b| >= |a|` 的相等边界、首尾 shift 强制 0、`localmin` 左严格右非严格、末元素双严格、第 0 列覆盖、`x < threshold` 严格、`argmax` 取首个 True、`argmin` 并列取首个、全 False 回退全局最小、`min_period + idx`、`f0 = sr / period`。
- mutation testing：40 个单点变异，要求**全部 CAUGHT**。这一轮踩到的坑值得记住：
  - 首轮只跑了 39 个变异，有 6 个存活。诊断发现两个是**真等价变异**（`shifts` 列表已初始化为 0.0，删掉「显式置首尾 0」的赋值并不改变结果），按既定原则必须剔除，改写成「让首尾参与抛物线计算」的形态才真正可测；
  - 另外三个存活变异（`localmin` 左严格 vs 非严格、末元素严格 vs 非严格）源于网格全是连续随机浮点，没有并列值，于是把网格改成四风格混合（连续、窄区间、量化到 1/3、0/1 二值）后补上；
  - 一个存活变异（`>=` vs `>`）需要 `|b| == |a|` 的精确相等情形，补了 `x = [-0.5, 0.0, 1.5]` 这个锚点。
- **教训沿用 836：mutation 全绿不能替代正确解自检。** 本轮每次改完用例都先跑一遍正确解的 `run`/`submit` 双模式自检，确认 `ACCEPTED` 之后再跑 mutation，避免「正确解其实错了、变异被假阳性判死」的情况。

## 不进判题域的边界

FFT 自相关（`audio.autocorrelate`）、`pyin` 的 HMM/Viterbi 概率解码与随机采样、`piptrack` 的谱峰估计、`__check_yin_params` 的异常分支、`center`/`pad_mode` 分帧、多通道广播、真实波形 IO 与重采样，全部不进判题域。本题只保留三段确定性的纯数值账。
