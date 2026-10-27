# 解题说明 · 预加重 / 分帧 / 加窗

## 为什么必须先预加重再切块

预加重是一阶 IIR（`y[t] = x[t] - a·x[t-1]`），只依赖相邻两个样本。如果先切块再逐帧加重，第 0 帧的第 0 个样本、以及每一帧的第 0 个样本都没有「前一个样本」可用，只能特殊处理成 `y = x`——于是每帧边界都被打断，帧移小于帧长时还会重复处理重叠区。正确写法是整条波形先过一遍，再切片。

## 帧数公式与尾巴

`F = 1 + (T - frame_len) // hop_len`。它有三个含义：第一帧从 0 开始、最后一帧的**起点**满足 `start + frame_len <= T`、不足一帧的尾巴丢弃。很多实现误用 `ceil`，会多出一帧越界的（被 pad 或崩溃）。

`hop_len > frame_len` 时相邻帧之间留空隙，公式一个字都不用改。

## 对称窗 vs 周期窗

同样叫 Hann，两套口径：

- 对称型（`sym=True`）：分母 `N-1`，两端严格为 0。用于滤波器设计、以及本题这种「把帧当独立信号分析」的场合。
- 周期型（`periodic=True`）：分母 `N`，等于对称型长度 `N+1` 去掉最后一点。用于 STFT（因为 DFT 默认信号是周期延拓的）。

`torch.hann_window(N)` 默认是**周期型**，`torch.hann_window(N, periodic=False)` 才是对称型。本题用对称型；同一批的 STFT 题用周期型，这就是最容易串味的一处。

## Hamming 端点不是 0

`0.54 - 0.46·cos(0) = 0.08`，所以 Hamming 每帧首尾保留 8% 的幅度，而 Hann 是 0。这个差异让 Hamming 的第一/最后一个样本不为零，是「端点断言」题的考点。

## 向量化写法

三行就够：`starts = arange(F) * hop_len`、`idx = starts[:, None] + arange(N)[None, :]`、`frames = y[idx] * w[None, :]`。用 gather 索引而不是 `unfold`，好处是 `hop_len > frame_len` 也自动正确（`unfold` 只支持带重叠的滑窗）。

## 复杂度

时间 O(T + F·N)，空间 O(F·N)（输出本身）。预加重是 O(T) 的一遍扫描，窗口只需预计算一次。

## 相关题目

- `audio-stft-frame`：分帧之后接 rfft，并把「对称窗 / 周期窗」的差异摆到台面上。
- `audio-mel-filterbank`、`audio-mfcc`：这条流水线后面的两级。
- `audio-resample-polyphase`：进这条流水线之前通常要先统一采样率。
