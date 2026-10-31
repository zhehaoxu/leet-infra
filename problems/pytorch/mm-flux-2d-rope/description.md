# 多模态：FLUX.1 二维 RoPE 频率表与相邻对旋转

## 题目描述

FLUX.1 的 transformer 把文本 token 与图像 token 拼成**一条联合序列**，并用**同一张** RoPE 表同时作用 $q$ 与 $k$。它的 head 维 $128$ 被切成三轴 $(16,56,56)$：轴 0 是帧（图像帧恒为 $0$，文本三轴全为 $0$），轴 1 是潜空间行号，轴 2 是潜空间列号。本题按 `huggingface/diffusers` 的 FLUX.1 口径搭出联合序列的编号 `ids` 与两张 float32 频率表，并对 `query` 做**相邻对**旋转。

参数、形状、返回值顺序、dtype 与异常约定**只在 starter 的双语契约里给出**；本题面只定义计算规则。

范围说明：只复现 FLUX.1 的**位置编号与频率表构造**，逐 head 复制与 $[B,S,H,D]$ 布局被折叠掉（同一张表对每个 head 相同，故把 head 维约去、只保留 $[B,S,D]$）；不含扩散模型权重、不含 latent 打包/解包、不含任何注意力打分计算。

## 网格编号

图像部分先搭 $(h,w,3)$ 的稠密编号，再行主序展平成 $(h\cdot w,3)$：

```text
image_ids = zeros(h, w, 3)
image_ids[..., 1] += arange(h)[:, None]      # 行号沿列方向广播
image_ids[..., 2] += arange(w)[None, :]      # 列号沿行方向广播
image_ids = image_ids.reshape(h * w, 3)      # row-major: k = row * w + col
text_ids  = zeros(text_len, 3)               # FLUX.1 把文本编号全置 0
ids       = cat([text_ids, image_ids], 0)    # 文本在前，联合序列共用一张表
```

轴 0 永远是 $0$（单帧图像），所以**每一行**的角度第 0 段都是 $0$。

## 每轴自己的频率

设第 $a$ 轴的维数为 $d_a$（本题 $d=(16,56,56)$，$\sum d_a=\text{head\_dim}$）。该轴有 $d_a/2$ 对，其逆频率与角度为

$$
\text{inv\_freq}_a[j]=\theta^{-2j/d_a},\qquad
\text{angles}_a=\operatorname{outer}\!\bigl(\text{pos}[:,a],\ \text{text{inv\_freq}}_a\bigr),\qquad j=0,\dots,d_a/2-1 .
$$

两处最容易写错：**指数分母是该轴自己的完整维数 $d_a$，不是 head_dim，也没有预先减半**；位置取的是 `ids.float()`（float32），而 `inv_freq` 与 `angles` 在 float64 里算。三个轴的 `outer` 结果沿末维拼接，得到每个 token $\sum d_a/2=\text{head\_dim}/2$ 个角度。

## 落表与旋转

每个轴的 $\cos/\sin$ 在 float64 里求值，随后沿末维 `repeat_interleave(2)` 把每个值复制成相邻两列，再 `.float()` 落到 float32，最后三轴沿末维拼接成 $(S,\text{head\_dim})$ 的两张表。因为复制，表满足

$$
\cos[2m]=\cos[2m+1],\qquad \sin[2m]=\sin[2m+1],
$$

所以旋转就是经典**相邻对**旋转，每对 $m$ 用自己的角度：

$$
\begin{pmatrix} q'_{2m}\\ q'_{2m+1}\end{pmatrix}=
\begin{pmatrix} c[m]&-\;s[m]\\ s[m]&c[m]\end{pmatrix}
\begin{pmatrix} q_{2m}\\ q_{2m+1}\end{pmatrix},\qquad
c[m]=\text{freqs\_cos}[\dots,2m],\ s[m]=\text{freqs\_sin}[\dots,2m].
$$

两张表恒为 float32（与 `query` 的 dtype 无关）；旋转在 `query` 自身的 dtype 下进行，`rotated` 的 dtype 跟随 `query`。

## 可验证的性质

- 帧轴恒 $0$ 且 $d_0=16$：每张表**前 16 列**对**所有行**都是 $\cos=1,\sin=0$，这 16 个分量永不旋转。
- 文本行的 `ids` 全零：整行 $\cos=1,\sin=0$，即 **FLUX.1 的文本 token 完全不旋转**；图像 $(0,0)$ 格同理。这是库的真实行为，不是本题的设定。
- $\theta=1$ 时所有 $\text{inv\_freq}=1$，角度退化为位置本身；位置为 $0$ 的行仍严格恒等。

## 样例 1：小轴维 $(2,4,4)$

展示值保留七位小数，均来自参考实现实跑。

```text
axes_dim = (2, 4, 4), theta = 10000.0, text_len = 1, image_grid = (1, 2)
S = 1 + 1*2 = 3, head_dim = 10, angles per token = 5
query = [[[1,0, 1,0, 1,0, 1,0, 1,0],     token 0: text
          [1,0, 1,0, 1,0, 1,0, 1,0],     token 1: image row 0, col 0
          [1,0, 1,0, 1,0, 1,0, 1,0]]]    token 2: image row 0, col 1

ids       = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]                       # float64 (3, 3)
inv_freqs = [1.0, 1.0, 0.01, 1.0, 0.01]
            axis0 (dim 2): [1.0] | axis1 (dim 4): [1.0, 0.01] | axis2 (dim 4): [1.0, 0.01]
angles    = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0.01]]  # float64 (3, 5)
freqs_cos = float32 (3, 10), row 2 = [1, 1, 1, 1, 1, 1, 0.5403023, 0.5403023, 0.99995, 0.99995]
freqs_sin = float32 (3, 10), row 2 = [0, 0, 0, 0, 0, 0, 0.8414710, 0.8414710, 0.0099998, 0.0099998]
rotated   = float32 (1, 3, 10)
            token 0 and token 1 are bit-exact copies of query
            token 2 = [1, 0, 1, 0, 1, 0, 0.5403023, 0.8414710, 0.9999500, 0.0099998]
```

## 样例 2：FLUX.1 默认 $(16,56,56)$

`query` 的每一行都是 $[1,0]$ 重复 $64$ 次，于是旋转结果的第 $2m$ 位就是该对的 $\cos$、第 $2m+1$ 位就是 $\sin$。

```text
axes_dim = (16, 56, 56), theta = 10000.0, text_len = 2, image_grid = (2, 3)
S = 2 + 2*3 = 8, head_dim = 128, angles per token = 64

ids (8, 3) float64:
[0, 0, 0]  token 0 (text)      [0, 0, 2]  token 4 (row 0, col 2)
[0, 0, 0]  token 1 (text)      [0, 1, 0]  token 5 (row 1, col 0)
[0, 0, 0]  token 2 (row 0, col 0)          [0, 1, 1]  token 6 (row 1, col 1)
[0, 0, 1]  token 3 (row 0, col 1)          [0, 1, 2]  token 7 (row 1, col 2)

inv_freqs (64,) float64, denominators are the axis dims:
  axis0 indices  0.. 7 (dim 16): 1.0, 0.3162278, 0.1, 0.0316228, 0.01, ...
  axis1 indices  8..35 (dim 56): 1.0, 0.7196857, 0.5179475, 0.3727594, ...
  axis2 indices 36..63 (dim 56): 1.0, 0.7196857, 0.5179475, 0.3727594, ...

angles (8, 64) float64:
  tokens 0..2 are all zeros
  token 4: angles[36] = 2.0                 # col 2 * inv_freqs[36]
  token 7: angles[ 8] = 1.0, angles[ 9] = 0.7196857, angles[36] = 2.0

freqs_cos / freqs_sin (8, 128) float32:
  every row, cols  0..15 -> cos 1.0, sin 0.0            # frame axis is always 0
  token 7, cols 16..21   -> cos 0.5403023, 0.5403023, 0.7520130, 0.7520130, 0.8688372, 0.8688372
  token 7, cols 16..21   -> sin 0.8414710, 0.8414710, 0.6591483, 0.6591483, 0.4950979, 0.4950979

rotated (1, 8, 128) float32:
  tokens 0, 1 and 2 are bit-exact copies of the input
  token 7, cols  0.. 3 -> 1.0, 0.0, 1.0, 0.0
  token 7, cols 16..21 -> 0.5403023, 0.8414710, 0.7520130, 0.6591483, 0.8688372, 0.4950979
  token 7, cols 36..39 -> 0.9993053, 0.0372673, 0.9996402, 0.0268237
```

## 约束

- `query` 为 CPU float32 或 float64，$(B,S,\text{head\_dim})$，$B\in[1,3]$，$S=\text{text\_len}+h\cdot w$；可非连续，不得被修改。
- $h,w\ge 1$，$\text{text\_len}\ge 0$，`axes_dim` 为三个正偶整数，$\theta\ge 1.0$；越界或非整数按 starter 契约抛 `ValueError`。
- 逐对判题容差：float32 输出 `atol=rtol=4e-5`，float64 输出 `atol=rtol=2e-12`，`ids` 要求逐位相等。
- 无梯度需求；`torch` 的向量化算子可自由使用，本题不禁止任何张量 API。
