# 解题说明 · vqllm 的 std 归一化 + 多步残差向量量化

## 一句话

先把每条 token 的向量除以它自己的无偏标准差，进"人人等尺度"的坐标系；然后拿若干套量化器接力，每套把当前残差切成 $N$ 段、每段在自己的子码本里取最近码字并减掉，跑 `num_residual_steps` 轮后交给下一套；重建时乘回尺度、key 侧再把交织过的通道顺序复原。本题额外要求把每一步的残差范数和带段偏移的下标全部记录下来。

## 为什么先除以每条 token 自己的标准差

码本是静态的、全体 token 共用，而分配步只做一件事：在 $M$ 条码字里找欧氏距离最近的一条。这个"最近"对绝对尺度毫不中立——幅值大两个数量级的 token 会整批落到码本上端，把高分辨率的码字吃光；幅值小的 token 只能落到同样偏大的码字上，相对误差爆炸。除以 $\sigma_t=\operatorname{std}(x_t)$ 之后，每条 token 都被拉回同一个尺度带，码本只需覆盖归一化后的那个流形。

尺度信息并没有丢：它变成每 token 一个浮点数（就是返回的 `scale`），重建时乘回去即可。所以 `error = ||最终残差|| * scale` 和 `residual_norm` 轨迹其实是同一个量在两个坐标系里的读数——轨迹在归一化坐标里算，`error` 换回原始量纲。

两个容易被顺手改掉的细节：`std` 是**无偏**的（`correction=1`，分母 $D-1$），而且上游只做除法、**不减均值**（`x = x / scale`）。前者影响所有 token 的尺度，后者决定了"常数 token 的 $z$ 并不是 0"（见下）。

## 残差链：$N$ 段独立、逐轮累减、逐套接力

一次量化 = $N$ 次彼此独立的最近邻：第 $k$ 段只和 `codebooks[q][k]` 的 $M$ 条码字比距离，比完把选中的 $N$ 条码字沿段轴拼回 $(B,L,D)$，再从残差里整体减掉。减完的残差进下一轮（同一套码本再来一遍），$S$ 轮跑完就换下一套，下一套接手时的残差就是上一套吐出来的——所以 `residual_norm[q][0] == residual_norm[q-1][-1]`。

四个自由度分工很清楚地对应四种"花 bit 的方式"：$\hat d$ 是一条码字管几维，$N$ 是把向量拆成几段，$M$ 是每段有多少候选，$S$ 与 $Q$ 是同一套复用几轮、总共接力几套。$S$ 和 $Q$ 都在削同一份残差，区别只是"重复用自己的码本"还是"换一套码本"。

下标空间是每套量化器自己的 $[0,N\cdot M)$：第 $k$ 段的结果加 $k\cdot M$。因此不同套之间重号是**正常**的，而 $M$ 还能逐套不同（实测第二套 $M=5$ 时偏移是 $0,5$ 而不是 $0,3$），测试里给的界也只是 `indices.max() < N * max(M)`。

## 通道重排让子码本看到跨通道的坐标

`view(B,L,N,dhat).transpose(-1,-2)` 把最后两维互换，于是第 $k$ 段拿到的不再是"连续 $\hat d$ 个通道"，而是"每个子段第 $k$ 个通道"这组跨通道的坐标：

```
view (N=3, dhat=4)            transpose -> (dhat, N)         flattened w
c0 c1 c2 c3 | c4 c5 c6 c7 | c8 c9 c10 c11
w[k*N+m] = c[m*dhat+k]
w = c0 c4 c8 | c1 c5 c9 | c2 c6 c10 | c3 c7 c11
segment 0 (sub-codebook 0): c0 c4 c8
segment 1 (sub-codebook 1): c1 c5 c9
segment 2 (sub-codebook 2): c2 c6 c10
```

动机是相关性结构：同一子码本的 $M$ 条码字本来是给"相邻 $\hat d$ 维"设计的，而相邻通道的尺度与相关性差异极大（少数特征通道吃掉大部分方差），一段连续通道往往同时属于"离群通道"和"普通通道"，一个码字要同时拟合两种量级。交织之后同一子码本负责的是整条向量上等间隔的坐标，跨通道取样让每段的分布更同质，也更贴近码本被训练时见到的分布。上游只在 key 上做这件事（value 的入口直接转调、没有任何重排）。

`split(dhat,-1) + stack(-1) + reshape` 是它的逆。复原和乘尺度可以交换顺序：`scale` 是 $(B,L,1)$、对通道置换不变，所以"先乘后复原"（上游的顺序）与"先复原后乘"（参考解的顺序）结果逐位一致。

## 五个易错口径

1. **无偏，而且不减均值**。写成 `sqrt(mean(z^2))` 会让 `scale` 整体偏小，`scale` 与 `quantized` 一起错；反过来，习惯"标准化"的人容易顺手 `x - x.mean(-1)`，那是另一个算法——本题和上游都只除尺度。实测 `x = [5,5,5,5]`：`scale` 由 0 换成 1，$z$ 仍是 $[5,5,5,5]$，重建 `[0.5, 1.0, 1.5, 3.0]`、`error = 7.245688`。
2. **零尺度精确换成 1**，不是 `clamp_min(eps)`。上游写的是 `torch.where(scale == 0, 1, scale)`；换成 eps 会让常数 token 的 $z$ 变成天文数字，整条链跟着炸。保证的是"不出现 NaN/Inf"，不是"重建为 0"。
3. **下标带 $k\cdot M$ 偏移**，$M$ 按当前这套量化器取；写回码本条目号要减 $k\cdot M$。
4. **同距取更小的条目号**。上游 `sim.argmax(dim=1)` 不定义并列顺序，本题钉死成最小下标，测试用 $\pm$ 对称条目和逐位重复的条目专门造出并列。稳妥写法是 `hit = dist == dist.min()` 再取 `hit` 里第一个 `True`（numpy 的 `argmax` 布尔掩码就是这个语义），别指望 `argmin` 的并列行为。
5. **轨迹的拼接点**：第 $q$ 套有 $S+1$ 个值，第 0 个是**接手时**、最后一个是**跑完 $S$ 轮后**；`error` 等于最后一套的末值乘 `scale`，`quantized` 用同一轮的 $z-\text{残差}$。

## 判分点

- 五个键的形状与 dtype：`indices` 恒为 int64，其余四个键（`scale`/`residual_norm`/`quantized`/`error`）一律跟输入 dtype（float32 进就得 float32 出）；
- 偏移与轨迹：`residual_norm` 首值就是 $\lVert z\rVert$，末值 × `scale` 就是 `error`，`std == 0` 时 `scale` 必须是 1 且全程无 NaN；
- reorder 的定向断言：$N=2,\hat d=2$ 时重排把 $[c_0,c_1,c_2,c_3]$ 变成 $[c_0,c_2,c_1,c_3]$，第一段变成 $(c_0,c_2)$，`indices` 必须随之改变，而 `scale` 两档一致、`quantized` 两档必须不同；
- 并列取小：$\pm$ 等距与重复码字两种局面都取最小条目号；
- 多套接力：`residual_norm[1][0] == residual_norm[0][-1]`，且第二套要把误差再压一个量级；
- 非法入参一律 `ValueError`（非三维、整型、空 list、二维码本、dtype 不符、$N\hat d\ne D$、`steps=0`、`steps` 是 bool、`reorder` 传 1 等）；
- 随机对拍：多组形状 × 多轮 × 多套 × 两档 reorder，逐键与一份**独立的 numpy oracle**（直接差方、不用展开式）比对；
- 入参不被就地修改：每个用例都先克隆、跑完再 `torch.equal` 比回去。

## 边界

- $D\ge 2$、$N\ge 1$、$M\ge 1$ 且 $N\hat d=D$。$N=1$ 是合法的（共享码本，$N$ 段退化成一段）；$\hat d=1$ 时重排就是恒等置换。
- **残差范数不保证逐轮下降**。最近邻最小化的是 $\lVert s-c\rVert$，而不是"让残差变小"；当码本里没有靠近原点的码字时，减掉一条码字反而会把残差变大。实测全零 token 配本题示例码本，轨迹是 $[0,\,0.25]$。测试的单调断言只作用在专门构造的嵌套码本上。
- `scale` 两档相等是数学事实，但浮点上重排后求和顺序变了，可能出现 $\sim 10^{-16}$ 的尾差；测试按 `_close` 的容差（float64 2e-9、float32 3e-5）比，不是 `torch.equal`，所以别为了"完全相等"去特殊处理 `scale`。
- 校验的先后顺序是有意的：先 `x` 的维数与 dtype、再 `reorder_channel`/`num_residual_steps`，然后才是 $N\hat d=D$，最后逐个 codebook 的形状与 dtype。于是 `D=1` 是被 "$D\ge2$" 这条拦下来的，不是被码本形状拦下来的。
- `codebooks` 只认 list/tuple：生成器、以及"套着普通 list 的 list"都会 `ValueError`（后者会在形状检查处被拒）。`x` 非连续或 `requires_grad=True` 都能正常算——全程出位运算加 `.detach()`，也就顺带满足了"不得修改入参"。

## 复杂度

设 $T=BL$。每套每轮每 token 要做 $N$ 段 × $M$ 条码字 × $\hat d$ 维的比较，即 $O(MD)$，总量 $O(T\,Q\,S\,M\,D)$；距离用展开式还是直接差方都是这一档，展开式只是把它写成一次 batched matmul。输出本身也是账的一部分：`residual_norm` 是 $Q(S+1)T$，`indices` 是 $T\,Q\,S\,N$ 个 int64。压缩侧的位宽账则是每 token 每套每轮 $N\log_2 M$ bit 的索引，加上每 token 一个 `scale` 浮点——码本容量 $M$ 和残差轮数 $S$ 是在同一份预算里换来换去。

## 出处

`iankur/vqllm`（arXiv 2410.15704，ENLSP-IV 2024；仓库 `README.md:3-4` 就给了这两个出处，题面引用与之一致）。以下 file:function:line 全部在本地 `/tmp/qsrc/vqllm` 逐行读过。

- `vqllm/quantizer/utils.py:22-25` `quantize()`：L23 `scale = x.std(dim=-1, keepdim=True)`、L24 `scale = torch.where(scale == 0, 1, scale)`、L25 `x = x / scale` —— 规则 1 的确切三行，无偏、只除不减均值、零换 1。
- `utils.py:41-53` `quantize()`：L44 外层 `for quantizer in quantizers:`、L48 内层 `for _ in range(quantizer.num_residual_steps):`、L49-51 `xq_, indices, commitment_loss_ = quantizer(x, ...)`、L52 `x = x - xq_` —— 规则 3 的"逐轮累减、逐套接力"。
- `utils.py:55` `x = (x_orig - x.detach()).to(x_dtype)` 与 `utils.py:58` `x = x * scale` —— 题面第 5 条的 `z - 最终残差`、再乘回尺度。（注意 `x_orig` 在 L38 采集，此时 L25 的除法已经做过，所以它本来就是归一化后的 $z$。）
- `utils.py:62-80` `quantize_key()`：L64 `reorder_channel = quantizers[0].reorder_channel`、L71-72 `x.view(*x_shape[:-1], -1, dim // codebook_entry_dim)` 后 `transpose(-1,-2).reshape(*x_shape)`、L77-78 `x.split(codebook_entry_dim, dim=-1)` 后 `torch.stack(x, dim=-1).reshape(*x_shape)` —— 规则 2 的重排与复原。
- `utils.py:83-85` `quantize_value()` 直接转调 `quantize`，全程不碰重排 —— 题面"上游只在 key 上用"核实为真。挂载点也印证：`vqllm/models/_modules.py:91-93` 对 k 走 `quantize_key`、L98-100 对 v 走 `quantize_value`（k/v 的形状是 `(b, s, n_kv, 1, head_dim)`，L84-86，正是"逐 token 一条向量"）；`vqllm/models/_component_builders.py:112` 只给 key 的量化器设 `reorder_channel`，L113-115 用 `nn.ModuleList([copy.deepcopy(quantizer_key) for _ in range(num_residual_codebooks)])` 把"多套量化器"表达成一个 list。
- `vqllm/quantizer/quantize.py:141-182` `VQVAEQuantize.decode()`：L146-147 `z.view(-1, num_codebooks, codebook_entry_dim)` + `permute(1, 2, 0)`、L150-152 码本 `view(num_codebooks, num_codebook_entries, codebook_entry_dim)`、L167-171 `sim = -(z.pow(2).sum(dim=1, keepdim=True) + codebook.pow(2).sum(dim=2, keepdim=True) - 2 * torch.matmul(codebook, z))`、L173 `indices = sim.argmax(dim=1).permute(1, 0)`、L176-178 `offset = torch.arange(self.num_codebooks) * self.num_codebook_entries` 再 `indices += offset`、L181-182 `z_q = self.codebook(indices).view(input_shape)` —— 规则 3/4 的展开式距离、逐段最近邻与 $k\cdot M$ 偏移。
- 校准侧（题面声明不在范围内，引用核实过）：`quantize.py:4` `from scipy.cluster.vq import kmeans2`、`quantize.py:188-209` `init_codebook()` 用 `kmeans2(..., minit="points")` 初始化码字、`quantize.py:9-46` `EmbeddingEMA` 的 `cluster_size_ema_update`/`embed_avg_ema_update`/`weight_update` 做 EMA 更新。
- 另有一条 triton 快路径 `vqllm/quantizer/quantize_kernel.py:181-208`：同样的逐套逐轮残差循环（L181-182）、同样的展开式欧氏距离（L200-206）、同样的 `idx += k * BLOCK_SIZE_C` 偏移（L208）；但 `quantize.py:69-70` 明确它只支持共享码本（$N=1$），`utils.py:27-36` 也要 `use_fast_quantizer` 才走。本题对的是 eager 分支。

**题面订正记录**：「约束」第 2 条原先写常数 token "换 1 后 $z$ 全 0，重建也全 0"，与题面自己的规则 1（`z = x / scale`，不减均值）以及上游 `utils.py:25` 都不一致——实测 `x=[5,5,5,5]` 时 $z=[5,5,5,5]$、重建 `[0.5,1.0,1.5,3.0]`、`error = 7.245688`；只有全零 token 才让 $z$ 真是 0，重建仍是被选中的码字（示例码本下 `[-0.05,-0.1,-0.1,-0.2]`，`error = 0.25`）。这条真正要保证的是 `scale` 恰为 1、不出现 NaN/Inf，双语题面已按此改写；`test_cases.py` 只钉 `scale == 1.0`。

**本题自己的补充**（上游没写、或做法不同，题面显式钉死的口径）：

- **并列顺序**：上游 `quantize.py:173` 的 `argmax` 不定义同距顺序，本题钉成"最小条目号"。
- **`indices` 是本题的产物**：`utils.py:49-53` 里 eager 分支拿到的 `indices` 直接被丢掉（只在训练时喂给 EMA），上游推理路径根本不返回下标；本题把它们堆成 $(Q,S,B,L,N)$ 的 int64。
- **`residual_norm` 轨迹是本题的仪器**：上游从不记录每轮前后的残差范数，$Q(S+1,B,L)$ 这个形状与"逐套首尾衔接"的约定都是本题的。
- **`error` 换了定义**：上游每 token 的量是 `commitment_loss`，即逐通道 MSE（`quantize.py:130-131`）；本题的 `error` 是 $\lVert\text{残差}\rVert_2\cdot\sigma$，不是 MSE，量纲差一个 $\sqrt D$ 级别。
- **内部精度是 float64**：上游内部一律 `.float()`（`utils.py:38,42`、`quantize.py:165-166`），本题按题面走 float64，再按输入 dtype 返回。
- **`num_residual_steps` 变成全局参数**：上游它是每个 quantizer 自己的属性（`utils.py:48` 读 `quantizer.num_residual_steps`），不同套本可以跑不同轮数；本题一个 int 管 $Q$ 套。
- **`reorder_channel` 变成显式入参**：上游读 `quantizers[0].reorder_channel`（`utils.py:64`），而且这个属性不在 `VQVAEQuantize.__init__` 里定义，是 `_component_builders.py:112` 事后挂到 key 的量化器上的；本题既不需要那个属性，也不再限定只能作用于 key。
- **码本作为显式入参**：上游码字存在 `EmbeddingEMA` 里、形状是扁平的 $(N\cdot M,\hat d)$，靠 `kmeans2` + EMA 生成；本题直接给每套一个 $(N,M,\hat d)$ 张量，训练侧全部剥离。
- **没有 mask**：上游 `quantize`/`decode`/kernel 处处带 `(B,L)` 的 `mask`（如 `quantize.py:88-89`、`quantize_kernel.py:164`），本题每条 token 一律量化。
- **只走欧氏**：上游有 `cosine_distance` 分支（`quantize.py:160-162`），但两个入口都硬编码 `cosine_distance = False`（`utils.py:63,84`），所以本题的"只用欧氏"是忠实而非改动。
- **校验与错误类型**：上游没有任何输入校验，$N\hat d\ne D$ 会以 `view` 的 RuntimeError 冒出来；本题把 $D\ge2$、dtype、非空 list、共享 $(N,\hat d)$、$M\ge1$、bool/int（bool 不算 int）全部钉成 `ValueError`。其中"$D\ge2$"纯粹是无偏 `std` 的推论，上游并未要求。
- **剥离训练分支**：`training`、`copy_grad`、直通梯度 `z_q = z + (z_q - z).detach()`（`quantize.py:137-138`）都不在本题；上游 eager 分支整段还包在 `torch.no_grad()` 里。
- **形状约定收紧成三维**：上游签名是 `(bsz, seq_len, *, dim)`，中间可以任意插 head 等维度（`quantize.py:104-105` 的 docstring）；本题固定 $(B,L,D)$，head 维折进 $D$ 或 $L$。

## 相关题目

- `quant-squeezellm-sensitive-kmeans`：同一条链的另一半——用敏感度加权的 Lloyd 把格点**学出来**；本题的码本是显式入参，只做推理侧的分配与残差累减。
- `quant-kivi-kv-asym`：也是"逐 token 一个 scale"，但分配是均匀仿射格点加 zero-point；本题的分配是在多套子码本里找最近码字，尺度之外还要存索引。
- `quant-double-quant-scale`：同样是两级结构，但第二级量化的是**第一级的 scale**（元数据本身）；本题的第二级量化的是**第一级的残差**，方向完全不同。
