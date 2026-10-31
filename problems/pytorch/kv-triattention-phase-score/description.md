# KV Cache：TriAttention 相位式重要性打分

## 题目描述

**TriAttention**（`WeianMao/triattention`，README badge 标注 arXiv:2604.04921）用「**相位相干性**」给 KV token 打分，而不是直接用 attention 权重。它先把已经过 RoPE 的 key **反解**回旋转前的复数表示，再与平均 query 做复数乘法：

\[
\tilde k = k_{\text{post}}\cdot e^{-i\theta},\qquad
r = q_{\text{mean}}\cdot \overline{\tilde k}
\]

\(r\) 的幅值给出「query 与 key 有多对齐」，相位 \(\varphi=\arg r\) 给出「方向差多少」。分数把幅值乘上 \(\cos(\Delta t\cdot\omega+\varphi)\)：前若干频率维再叠加一个由几何偏移 \(\{1,2,4,\dots\}\) 得到的时间相位 \(\Delta t\cdot\omega\)，其余频率维只看空间相位。最后逐 `(layer, head)` 在 token 维做 z-normalize，跨层取均值，每个 head 取 top-k。

## 规则

实现 `solve(q_mean, q_abs, k, rope_cos, rope_sin, omega, pos, keep, current_end_frame, frame_seq_length=1, temporal_count=None, offset_max_frames=128, score_aggregation="mean", normalize_scores=True)`：

### 入参

- `q_mean`：\((L,H,F,2)\)，每个 `(layer, head, freq)` 的平均 query 复数，最后一维是 `(实部, 虚部)`。
- `q_abs`：\((L,H,F)\)，同一位置上的 query 幅值基准，用来补一个与相位无关的加性项。
- `k`：\((L,T,H,2F)\)，**已过 RoPE** 的 key；最后一维按 `(实部, 虚部)` 相邻打包，第 \(f\) 个频率对应下标 \(2f,2f+1\)。
- `rope_cos`、`rope_sin`：\((T,F)\)，每个 token、每个频率的 \(\cos\theta\)、\(\sin\theta\)。
- `omega`：\((F,)\)，每个频率的角速度。
- `pos`：\((T,)\) 的整数张量（不能是浮点）。
- `keep`：`int`，每个 head 保留的 token 数，\(0\le \text{keep}\le T\)。
- `current_end_frame`：`int`，当前窗口结束帧。
- `frame_seq_length`：`int >= 1`，一个 frame 覆盖多少 token，默认 1。
- `temporal_count`：`int`，前多少个频率维带时间相位；`None` 时取
  \[
  F-2\lfloor F/3\rfloor .
  \]
- `offset_max_frames`：`int >= 0`，几何偏移的上界，默认 128。
- `score_aggregation`：`"mean"`（默认）或 `"max"`，在偏移维上聚合。
- `normalize_scores`：`bool`，是否做逐 `(layer, head)` 的 z-normalize，默认 `True`。

所有浮点入参必须同 dtype（float32 / float64）；`L/H/F/T` 必须一致；`keep`、帧长、偏移上界等必须是 `int`（`bool` 不接受）。形状、dtype、取值范围不合法时抛 `ValueError`。

### 计算

1. **反解 RoPE**：把 `k` reshape 成 \((L,T,H,F,2)\)，对每个频率
   \[
   \tilde k_{\text{re}}=k_{\text{re}}\cos\theta+k_{\text{im}}\sin\theta,\qquad
   \tilde k_{\text{im}}=k_{\text{im}}\cos\theta-k_{\text{re}}\sin\theta .
   \]
2. **相位与幅值**：
   \[
   \varphi=\operatorname{atan2}\!\left(\Im(q_{\text{mean}}\overline{\tilde k}),\,
   \Re(q_{\text{mean}}\overline{\tilde k})\right),
   \]
   \[
   A=|q_{\text{mean}}|\,|\tilde k|,\qquad
   E=(q_{\text{abs}}-|q_{\text{mean}}|)\,|\tilde k| .
   \]
3. **几何偏移**：取 \(\{1,2,4,\dots\}\) 中所有 \(\le\) `offset_max_frames` 的值；若一个都没有（`offset_max_frames=0`），退化为 \(\{1\}\)。设 token \(t\) 所在帧为 \(\lfloor pos_t/\text{frame\_seq\_length}\rfloor\)，则
   \[
   \Delta t_{t,o}=\text{current\_end\_frame}-\left\lfloor \frac{pos_t}{\text{frame\_seq\_length}}\right\rfloor+o .
   \]
   输出 `delta_t` 形状 \((T,O)\)，\(O\) 是偏移个数。
4. **时间掩码**：前 `temporal_count` 个频率取 1，其余取 0；记作 \(m_f\)。
5. **逐偏移分数**：
   \[
   b_{l,h,t,o}=\sum_f A_{l,h,t,f}\cos\!\big(\Delta t_{t,o}\,m_f\,\omega_f+\varphi_{l,h,t,f}\big),
   \]
   \[
   \text{score}_{l,h,t}=\operatorname*{mean}_o\Big(b_{l,h,t,o}+\sum_f E_{l,h,t,f}\Big).
   \]
   `score_aggregation="max"` 时把 \(\operatorname*{mean}_o\) 换成 \(\max_o\)；加性项 \(\sum_f E\) 不随偏移变化。
6. **归一化**：`normalize_scores=True` 时
   \[
   \text{normalized}_{l,h,t}=\frac{\text{score}_{l,h,t}-\mu_{l,h}}
   {\max(\sigma_{l,h},\,10^{-6})},
   \]
   其中 \(\mu,\sigma\) 在 token 维上计算，\(\sigma\) 用**总体标准差**（`unbiased=False`）。
7. **跨层平均**：`head_score = normalized.mean(dim=0)`，形状 \((H,T)\)。
8. **逐 head top-k**：按 `head_score` 降序取前 `keep` 个（同分取更小 token 下标），再把下标升序排列。

### 返回值

返回 6 键字典：

- `phi`：\((L,H,T,F)\)，相对相位 \(\varphi\)；
- `delta_t`：\((T,O)\)，几何偏移后的帧差；
- `score`：\((L,H,T)\)，聚合后的原始分数；
- `normalized`：\((L,H,T)\)，z-normalize 后的分数（关闭归一化时等于 `score`）；
- `head_score`：\((H,T)\)，跨层平均后的每 head 分数；
- `kept`：\((H,\text{keep})\) int64，每行升序、无重复。

除 `kept` 外全部沿用输入 dtype；内部用 float64 计算，不得就地修改任何输入。`keep=0` 返回形状 \((H,0)\) 的 `kept`。

## 示例

```
L=1, H=1, F=2, T=2
q_mean = [[[[1, 0], [0, 0]]]]        # (1,1,2,2)
q_abs  = [[[1, 0]]]                  # (1,1,2)
k      = [[[[1, 0, 1, 0]],           # token 0: freq0=(1,0), freq1=(1,0)
           [[0, 1, 1, 0]]]]          # token 1: freq0=(0,1), freq1=(1,0)
rope_cos = ones(2, 2), rope_sin = zeros(2, 2)
omega = [1.0, 0.0], pos = [0, 1]
keep = 1, current_end_frame = 2, temporal_count = 1, offset_max_frames = 2
（其余取默认：frame_seq_length=1, score_aggregation="mean", normalize_scores=True）

phi        = [[[[0.0, 0.0], [-1.5707963267948966, 0.0]]]]     # (1,1,2,2)
delta_t    = [[3.0, 4.0], [2.0, 3.0]]                          # (2,2)
score      = [[[-0.8218180587320287, 0.5252087174427744]]]     # (1,1,2)
normalized = [[[-1.0, 1.0]]]                                   # (1,1,2)
head_score = [[-1.0, 1.0]]                                     # (1,2)
kept       = [[1]]                                             # (1,1) int64
```

`offset_max_frames=2` 给出偏移 \(\{1,2\}\)：token 0 的 \(\Delta t=2+\{1,2\}=\{3,4\}\)，token 1 为 \(1+\{1,2\}=\{2,3\}\)。`temporal_count=1` 让 freq 0 带时间相位、freq 1 冻结。token 0 的 \(\varphi=0\)、幅值 1，得分 \((\cos 3+\cos 4)/2=-0.821818\ldots\)；token 1 的 \(\varphi=-\pi/2\)，得分 \((\cos(2-\pi/2)+\cos(3-\pi/2))/2=0.525208\ldots\)。两个 token 做 z-normalize 后正好是 \([-1,1]\)，`keep=1` 保留分数更高的 token 1。

## 约束

- CPU float32 / float64；所有浮点入参同 dtype；`q_mean` 最后一维必须是 2，`k` 最后一维必须是 \(2F\)。
- `pos` 必须是一维整数张量；`keep`、`current_end_frame`、`frame_seq_length`、`offset_max_frames` 必须是 `int`；`normalize_scores` 必须是 `bool`。
- `0 <= keep <= T`；`frame_seq_length >= 1`；`offset_max_frames >= 0`；`0 <= temporal_count <= F`；`score_aggregation` 只能是 `"mean"` / `"max"`。
- 偏移集合是 \(\{1,2,4,\dots\}\) 中不超过 `offset_max_frames` 的值；为空时用 \(\{1\}\)。
- z-normalize 的 \(\sigma\) 用总体标准差并钳到 \(10^{-6}\)：整行常数（包括 \(T=1\)）归一化后为 0，不能出现 NaN / Inf。
- 同分一律取更小的 token 下标；`kept` 每行升序且不重复。
- 非法形状、dtype、非整数 `pos`、越界的 `keep` / `temporal_count`、不支持的聚合方式等均抛 `ValueError`。
