# 解题说明 · PolarQuant 的极坐标成对量化

## 一句话

把 post-RoPE 的 key 沿「前后两半」折成 $D_h$ 个二维向量，换成 $(\phi, r)$；角度和半径各自在 token 组内做 min-max 向下取整量化，两个码拼成一个整数 `indices = (rho << tbits) + theta`，重构时都取 bin 中心再用 $\cos/\sin$ 摊回 $D$ 维。

## 为什么是「前后两半」而不是相邻维

RoPE 的实现口径决定了配对方式。`rotate_half` 把 head_dim 切成两半再拼：$\text{rot}(x) = [\,-x_{D_h:},\ x_{:D_h}\,]$，也就是旋转矩阵作用在 $(x_d,\ x_{D_h+d})$ 这一个平面上。上游的 `key_states.view(B, Nk, Nb, G, 2, D)` 里那个 `2` 就压在这两半之间，`phi = atan2(x[..., 1, :], x[..., 0, :])` 取的是**次半维作 $y$、首半维作 $x$**。若按相邻维 $(x_{2d}, x_{2d+1})$ 折，等于换了一套旋转基，$\phi$、$\theta$、`recon` 全错——`check_pairing_is_halves_not_interleaved` 就是拿 $(2,5)$ 这一对来卡这个分歧的。

## 分母 $2^{\text{bits}}$、floor、夹回：三件事得连起来看

上游写的是 `floor((phi - tmn) / tscale)`，`tscale = (tmx - tmn) / 2**tbits`。于是最大元的商恰好是 $2^{\text{bits}}$，`.to(uint8)` 之后再由 `clamp(0, 2**bits-1)` 压回顶格。也就是说「最大元能表示」是**夹回给的**，不是分母给的；把分母写成 $2^{\text{bits}}-1$（仿射量化的常见习惯）反而让最大元落在 $2^{\text{bits}}-2$，整组码字平移一格。`check_max_lands_on_top_code_and_centre_is_inside_bin` 用 `tbits ∈ {2,3,5}` 三档同时钉住「最大元顶格、最小元为 0、重构角落在组内」。

重构取 $(\text{code}+0.5)\cdot\text{scale}+\min$ 是上游参考核里显式写出的：它先构造 `pphi = tscale * arange(2**tbits) + 0.5*tscale + tmn` 再查表，所以是 bin 中心而不是下边界。这一条也解释了为什么极坐标量化不会把最大角重构到 $\max$——它重构到 $\max - 0.5\cdot\text{scale}$。

## 折负角会改分箱

`atan2` 给 $(-\pi,\pi]$，`where(phi < 0, phi + 2π, phi)` 把它搬到 $[0,2\pi)$。这一步不只是好看：折回来之后组内最小值可能换成另一个 token 的角，$\theta$ 全体平移。`check_negative_angles_wrap_and_change_the_min` 里 $2\pi-0.1$ 折成 $\approx 6.18$ 后组最小变成 $0.0$，码字序列从「按原角排」变成 $[1,3,0,2]$；同一条测试还确认 `atan2(0, 3) = 0` 不会被误折成 $2\pi$。

## 本题自己的四处口径（与上游不同，已在题面写明）

- 上游把码字 `.to(torch.uint8)` 再 `clamp`，位宽超过 8 会静默截断；本题全程 int64 + 显式 `clamp(0, 2**bits-1)`，所以 `rbits/tbits` 可以放心给到 6、7。
- 上游对 `tscale`/`rscale` 无条件相除，组内恒定时得到 $0/0=$ NaN；本题定义 `where(scale > 0, floor(...), 0)`，常数组每个元素质字取 $0$。因为 bin 中心公式在 $\text{scale}=0$ 时退化成 $(0+0.5)\cdot 0 + \min = \min$，常数组反而**精确重构**，`check_constant_group_reconstructs_exactly` 断言 `recon == keys` 到 $10^{-12}$。
- 上游 `rbits=tbits=4`、`group_size=128` 写死在类属性里；本题把它们提到入参，$r_{\text{bits}}$ 只管半径、$t_{\text{bits}}$ 只管角度和打包位移。
- 上游打包成 uint8；本题 `indices` 返回 int64，但仍严格是 `(rho << tbits) + theta`，因此上界 $2^{\text{rbits}+\text{tbits}}-1$（`check_random_grid_and_dtype_against_oracle` 会验这条不变量）。

## 判分点

键必须恰好 6 个；`phi/radii/theta/rho/indices` 形状 $(B,H,L/G,G,D_h)$、`recon` 形状 $(B,H,L,D)$；三个码字键是 int64，两个几何键与 `recon` 跟随输入 dtype。内部 float64，所以 float32 入参与 float64 入参给出的**码字完全相同**（`check_random_grid_and_dtype_against_oracle` 直接比 `theta.tolist()`），只有 `recon` 差一个 float32 容差。入参只读。

## 边界

$D=2$（$D_h=1$）与 `rbits=tbits=1` 都合法；`group_size = L` 时只有一组。$L$ 不是 `group_size` 的整数倍、$D$ 为奇数、`keys` 非四维或 dtype 不是 float32/float64、三个参数里出现 `bool`/浮点/非正整数一律 `ValueError`——注意 `L=0` 也算非法（测试用 `torch.zeros(1,1,0,4)` 卡），别写成「$0$ 是任意正整数的倍数」而放行。

## 复杂度

$O(B\cdot H\cdot L\cdot D_h)$ 次三角与比较，逐组 min/max 与 argmin 都是组内线性；打包是两次整数位运算，不额外开 $2^{\text{bits}}$ 的码本表。

## 出处

`PolarQuant` 仓库（作者自带实现），逐行对照：
- `models/modeling_llama_polar.py` L135-157，`LlamaPolarGroupAttention.quantize_and_pack_nbit`（类属性 `rbits=4, tbits=4, group_size=128, residual_length=128`，解码走 triton 核）；
- `models/kernel4group.py` L113-158，`__main__` 下作者标注为 "Code Below is the pytorch implementation of our kernel" 的参考段——`atan2` 与负角折回、`tscale/rscale` 的分母、`floor + clamp`、`pphi/rradii` 的 $+0.5$ 中心、`indices = (rho << tbits) + theta` 都在这里。

仓库 README 只写 NeurIPS 2025，**没有给出 arXiv 编号**，所以本题按纪律以代码路径为证据；awesome list 流传的 2502.02617 未经仓库核实，不作为出处。

## 相关题目

`quant-kivi-kv-asym`（KV 的通道/token 混合精度非对称量化）与 `quant-vqllm-residual-vq`（逐 token 尺度归一 + 残差码本）都在压 KV，但都按标量通道量化；本题换的是**表示坐标系**（角度×半径），并把两个码拼进同一字。`kv-triattention-phase-score` 同样把 RoPE 后的 key 还原成相位，但用来打分驱逐而不是量化。
