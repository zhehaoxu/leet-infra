# 题解：FLUX.1 二维 RoPE 频率表与相邻对旋转

## 真实来源与范围区别

本题背景是 **FLUX.1 的二维 RoPE 位置编码**，口径取自 `huggingface/diffusers` 的 `main` 分支（2026-10-22 拉取核对；本机评测环境未安装 diffusers，行号以上游文件为准）：

- `src/diffusers/models/transformers/transformer_flux.py` 第 500–526 行：`FluxPosEmbed`，文件头注明 “modified from black-forest-labs/flux .../src/flux/modules/layers.py#L11”。它按 `ids.shape[-1]` 逐轴调用 `get_1d_rotary_pos_embed(..., theta=self.theta, repeat_interleave_real=True, use_real=True, freqs_dtype=float64)`，把各轴的 cos/sin 沿末维拼接。
- 同文件第 749–750 行：`ids = torch.cat((txt_ids, img_ids), dim=0)` 后 `image_rotary_emb = self.pos_embed(ids)` —— **整条联合序列一张表**，第 118–119 行 `apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)` 对 q 与 k 用同一张表，不做 img/text 拆表。
- `src/diffusers/models/embeddings.py` 的 `get_1d_rotary_pos_embed`（flux / hunyuan-dit / cogvideox 分支）与 `apply_rotary_emb`。
- `src/diffusers/pipelines/flux/pipeline_flux.py` 第 508–519 行：`_prepare_latent_image_ids` 的 `latent_image_ids = torch.zeros(height, width, 3)` 加行/列两路 `arange` 广播、再 `reshape(height * width, 3)`；文本侧 `text_ids = torch.zeros(prompt_embeds.shape[1], 3)`。
- 模型默认 `axes_dims_rope=(16, 56, 56)`、`theta=10000`、`attention_head_dim=128 = sum(axes_dims_rope)`。

**只复现 FLUX.1 的构造，且刻意折叠了实现细节：**

- 上游是 $[B,S,H,D]$（$H$ 个注意力头共享同一张表），本题把 head 维约掉只留 $[B,S,D]$，这是有意的简化，不是上游张量布局的复刻。
- 不涉及任何扩散模型权重、不涉及 `_pack_latents` / `_unpack_latents` 的潜空间打包解包、不做注意力打分，也不算 $q/k$ 之外的任何分支。
- 上游 `FluxPosEmbed` 会把 `freqs_dtype` 按设备下调（CPU 上为 float64），本题固定按 CPU float64 口径搭表；多帧/视频、NTK 缩放 (`ntk_factor`)、线性缩放 (`linear_factor`) 均不在本题范围内。

## 与 rope-embedding、mm-mrope-position-ids 的区别

本题既**不是** `rope-embedding`（order 34），也**不是** `mm-mrope-position-ids`（order 228），三点差异是判题口径的核心：

1. `rope-embedding` 只有**单一轴**：整条 head 维共用一组 $\theta_i=\text{base}^{-2i/d}$，$d$ 就是 head_dim，角度来自 token 的一维序号。本题把 head_dim **切成三轴** $(16,56,56)$，每轴的指数分母是**该轴自己的维数**（`arange(0, dim, 2) / dim`，且 $dim$ 是完整轴维、没有预先减半），位置来自 $(t,h,w)$ 三个不同分量；轴维一改（如 $(4,4,4)$），频率随之改变，而单轴实现的频率只跟 head_dim 有关。
2. `mm-mrope-position-ids` 只做**整数 $(t,h,w)$ 位置编号的构造**，没有任何三角函数与旋转。本题的编号来自 FLUX 的 `_prepare_latent_image_ids` 布局（帧轴恒 0、文本三轴全 0、行主序展平），并且要继续搭出频率表、角度表并完成相邻对旋转。
3. 本题的表是**先 float64 计算、`repeat_interleave(2)` 复制、再 `.float()` 落 float32**；单轴 RoPE 题全程在输入 dtype 里算，也没有“每对角度占两列”这一层。

## 上游数值链与本题精度

```text
# get_1d_rotary_pos_embed, the flux / hunyuan-dit / cogvideox branch
freqs     = 1.0 / (theta ** (arange(0, dim, 2, dtype=freqs_dtype) / dim))   # dim = FULL axis dim
freqs     = outer(pos, freqs)                                               # (S, dim/2), float64 on CPU
freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()                 # (S, dim), float32
freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
# apply_rotary_emb with use_real=True and sequence_dim=1, x is [B, S, H, D]
x_real, x_imag = reshape(x, ..., -1, 2).unbind(-1)                          # adjacent pairs
x_rotated = stack([-x_imag, x_real], dim=-1).flatten(last two dims)
out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
```

`repeat_interleave(2)` 把每个值复制成相邻两列，于是 $\cos[2m]=\cos[2m+1]$：这张“二维表”其实就是经典相邻对旋转，每 token $\sum d_a/2=64$ 个角度。

本题的精度口径（写进 starter 契约，判题按此对拍）：

- **表恒 float32**：`angles` 与其余三项返回值是 float64，`freqs_cos` / `freqs_sin` 无论 `query` 是什么 dtype 都是 float32，因为上游就是 `.float()` 落表。
- **旋转在 query dtype 下做**：`float32` 输入全程 float32；`float64` 输入让 float32 表被提升到 float64 后相乘，结果仍是 float64。这里与上游 `apply_rotary_emb` 的 `x.float()` **有意不同**——上游会把 float64 的 $x$ 降到 float32 再算，而本题保留 query 自身精度，为的是能用 float64 纯标量 oracle 以 `2e-12` 对拍；`rotated` 的 dtype 始终等于 `query` 的 dtype。
- 位置照抄上游的 `pos = ids.float()`（先过一遍 float32），但返回的 `ids` 是 float64。测试里的小整数位置在 float32 下精确，因此不影响判题。

## 推导与实现

1. 先按 starter 契约校验：`axes_dim` 是三个正偶整数（bool 与浮点非法，长度必须为 3）、`theta` 是有限实数且 $\ge 1.0$、`image_grid` 两项都是 $\ge 1$ 的整数、`text_len` 是非负 Python int（bool 非法）、`query` 恰三维且末维等于 `sum(axes_dim)`、$S=\text{text\_len}+h\cdot w$。任一不满足抛 `ValueError`。
2. 搭 `ids`：`zeros(h, w, 3)` 的第 1 轴加 `arange(h)[:, None]`、第 2 轴加 `arange(w)[None, :]`，行主序 `reshape(h*w, 3)`；文本是 `zeros(text_len, 3)`；`cat([txt, img], dim=0)`，文本在前。
3. 逐轴搭频率：`inv_freq = theta ** (-arange(0, d_a, 2, dtype=float64) / d_a)`，`angles = outer(pos[:, a], inv_freq)` 在 float64 里得到 $(S, d_a/2)$。
4. 每轴 `cos/sin` 用 float64 求值、`repeat_interleave(2, dim=-1)` 复制、`.float()` 落表，三轴沿末维拼接成 $(S,\text{head\_dim})$。
5. 旋转按相邻对：取 `freqs_cos.reshape(S, -1, 2)[..., 0]` 作为 $c[m]$（与取 `[..., 2m]` 等价，因为两列相同），`out[2m]=q[2m]c[m]-q[2m+1]s[m]`、`out[2m+1]=q[2m+1]c[m]+q[2m]s[m]`，用 `stack(...).reshape(...)` 收尾，保持 `query` dtype 并支持非连续输入。
6. 返回顺序固定为 `(rotated, freqs_cos, freqs_sin, ids, angles, inv_freqs)`。

FLUX.1 由此得到两个真实且常被误传的属性：文本 ids 全零 $\Rightarrow$ 文本 token **完全不旋转**；又因帧轴恒零且 $d_0=16$，**每张表前 16 列对所有行都是 $\cos=1,\sin=0$**。$\theta=1$ 时所有 $\text{inv\_freq}=1$，角度退化为位置本身（并非“恒等”），位置为 0 的行才严格恒等。

## 独立 oracle 与实跑

判题 `_oracle` 是纯 Python 标量实现：嵌套 `for` 循环按 $(row,col)$ 生成网格 ids，逐对用 `math.pow` 现算 `inv_freq`、逐对用 `math.cos` / `math.sin` 现算三角值、手工 `extend((v, v))` 做 `repeat_interleave` 的复制语义、再逐对做标量 $2\times2$ 旋转；不使用 `torch.outer`、`repeat_interleave` 或张量 `cos/sin`。`math` 的 float64 结果与 torch CPU float64 结果同库，落 float32 用 `struct.pack/unpack` 复刻 `.float()` 的舍入，因此 float64 用例能压到 `2e-12`，float32 用例按同族约定用 `4e-5`。

本地 PyTorch **2.14.1** 上，`add_problem.py --selftest mm-flux-2d-rope --mode submit` 得到 **15/15 ACCEPTED**，`--mode run` 得到 **2/2 ACCEPTED**（两个样例用例标了 `example = True`）。覆盖：两个实跑样例、多组手写配置对拍、10 组随机种子扫描（变 $B,h,w,\text{text\_len},axes\_dim,\theta$ 与 dtype）、前 16 列恒等、文本行整体恒等、$\theta=1$ 退化、$1\times1$ 网格、`text_len=0`、$(4,4,4)$ 与 $(2,4,6)$ 的每轴分母证明、`cos[2m]==cos[2m+1]` 复制、float32/float64 双 dtype 的落表策略、非连续输入、不修改输入，以及全部 `ValueError` 分支。

题面两个样例的每个数值都取自参考实现实跑：

```text
axes_dim=(2, 4, 4), theta=10000.0, text_len=1, image_grid=(1, 2), query rows = [1,0] x 5
ids[2]      = [0.0, 0.0, 1.0]
angles[2]   = [0.0, 0.0, 0.0, 1.0, 0.01]
freqs_cos[2]= [1, 1, 1, 1, 1, 1, 0.5403022766113281, 0.5403022766113281, 0.9999499917030334, 0.9999499917030334]
rotated[0,2]= [1, 0, 1, 0, 1, 0, 0.5403022766113281, 0.8414709568023682, 0.9999499917030334, 0.009999833069741726]

axes_dim=(16, 56, 56), theta=10000.0, text_len=2, image_grid=(2, 3), query rows = [1,0] x 64
ids[7]       = [0.0, 1.0, 2.0]
inv_freqs[1] = 0.31622776601683794        # axis0, 10000**(-2/16)
inv_freqs[9] = 0.719685673001152          # axis1, 10000**(-2/56)
angles[7][8] = 1.0, angles[7][36] = 2.0
rotated[0,7][16:22] = [0.5403022766113281, 0.8414709568023682, 0.7520129680633545,
                       0.6591483354568481, 0.8688372373580933, 0.4950978755950928]
rotated[0,0] 与 rotated[0,2] 逐位等于输入（文本行与 (0,0) 格不旋转）
```

这些数值是本题参考实现的运行记录，不是从任何 FLUX.1 权重或完整管线前向里摘出来的输出。

## 复杂度

搭表 $O(S\cdot \text{head\_dim})$（其中三角函数 $O(S\cdot \text{head\_dim}/2)$），旋转 $O(B\cdot S\cdot \text{head\_dim})$；额外空间为 $ids$、$angles$ 与两张表的 $O(S\cdot \text{head\_dim})$ 加上输出张量。
