# 详解：结构化丢弃：DropPath 与 Dropout2d/3d

## 出处

- Stochastic Depth：Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016（**arXiv:1603.09382**）——训练时以概率把整条残差支路置为恒等（逐样本 Bernoulli 硬币），存活概率随深度线性衰减；DropPath 是该机制的最小实现单元：单层「整样本丢弃 + 重缩放」。
- timm 的 DropPath 实现口径（huggingface/pytorch-image-models，`timm.layers` 的 drop_path）：keep_prob = 1−p，mask = floor(keep_prob + rand\((B,1,\dots,1)\))，y = x.div(keep_prob) · mask——与本题 \(y=x\cdot mask/(1-p)\) 同一公式；差别只在 mask 由 RNG 采样（timm）还是显式传入（本题）。
- Dropout2d / Dropout3d：torch 官方文档——"Randomly zero out entire channels"（每个样本的每张特征图独立置零，通道是 2D/3D 特征图 \(\text{input}[i,j]\)）；torch 文档同时引用 Tompson et al., *Efficient Object Localization Using Convolutional Networks*（2015，**arXiv:1502.02604**）作为 spatial dropout 的出处：特征图内相邻像素强相关时，i.i.d. 丢弃不起正则作用、只等效降低学习率，须整图丢弃。
- inverted scaling 是 dropout 家族训练态的标准记账（与 `F.dropout` training=True 一致）：若 mask ~ Bernoulli(\(1-p\))，则 \(E[y]=x\cdot(1-p)/(1-p)=x\)，推理态无需再缩放。

## 完整推导与下标约定

**一份公式、三种粒度**。记 \(x\ (N,C,S_1,\dots,S_k)\)，统一式 \(y[b,c,\dots]=x[b,c,\dots]\cdot m[b,c]/(1-p)\)，三种 mode 只是 \(m\) 的广播形状不同：

- droppath：\(m\) 形状 \((B,1,\dots,1)\)——每样本一个硬币，样本内全部元素同生共死（timm 的 per-sample stochastic depth）；
- dropout2d：\(m\) 形状 \((N,C,1,1)\)——每样本每通道一个硬币，整张 \(H\times W\) 特征图置零；
- dropout3d：\(m\) 形状 \((N,C,1,1,1)\)——通道块含深度维，\((D,H,W)\) 整块置零。

被丢块严格为 0（\(0\cdot x=0\)，无近似），存活块恰为 \(x/(1-p)\)。mask 允许两种等价形式——紧凑形 \((B,)\)/\((N,C)\) 与满 1 形 \((B,1,\dots,1)\)/\((N,C,1,\dots,1)\)；实现时 reshape 成满 1 形再乘即可，两种形式 `torch.equal` 级等价（判题断言）。

**形状白名单为何是「广播后恰为 x 形状」**：广播语义下 \((1,C)\)、\((B,1)\) 等也能乘，但那意味着「跨样本共享通道硬币」「通道内再分裂」等别的粒度——本题把粒度钉死，任何额外/缺失维都抛 `ValueError`。droppath 对秩 \(\ge1\) 的任意尾部形状通用（\((B,)\) 恒可广播出 \(1\) 尾巴）；2d/3d 则要求恰好 4/5 维，秩不符先行拒绝。

**p 的边界账**：p=0 → scale=1，\(y=x\cdot mask\)；全 1 mask 时是**严格恒等**（`torch.equal` 要求逐位一致——实现不能走任何会改 dtype 或引入数值噪声的路径）。\(p\ge1\) → \(1/(1-p)\) 除零或为负，语义不存在；\(p<0\) 与非有限 p（nan/inf）同理——nan 的大小比较恒为 False，必须用 isfinite 显式判。`F.dropout` 的合法域同为 \([0,1)\)。

**期望记账与「谁缩放」**：把 mask 看作 Bernoulli(\(1-p\)) 的实现时 \(E[y]=x\)——这是 inverted（训练期除以存活率）与旧式「推理期乘 \((1-p)\)」的分工。mask 显式给定后公式退化为确定性逐元素缩放，但**缩放约定仍按训练口径执行**：scale 只由 p 决定、与 mask 内容无关（判题按 \(x/(1-p)\) 断言存活块）。

**bool mask 语义**：True=1=保留、False=0=丢弃。按 starter 约定先 `.to(x.dtype)` 再乘——与 0/1 浮点 mask 走完全同一路径（判题断言两类型 `torch.equal`），同时保证输出 dtype 与 x 一致。

## 核心与易错点

1. **粒度写成逐元素**：把 mask 弄成 x 的完整形状逐元素乘是最常见错误——判题的「被丢块整体为 0 / 存活块整体 \(x/(1-p)\)」按块检查。
2. **缩放漏乘或乘错侧**：\(y=x\cdot mask\)（漏 scale）、\(y=x\cdot mask\cdot(1-p)\)、\(y=x\cdot mask/p\) 全错；scale \(=1/(1-p)\) 只由 p 决定。
3. **mask 形状白名单**：droppath 的 3 维 x 只接受 \((B,)\)/\((B,1,1)\)；\((B,1,1,1)\)、\((B,C)\)、\((B,1)\) 都抛 `ValueError`。2d/3d 的 \((N,C,1)\) 中间形同样非法。
4. **p 校验含非有限值**：`float("nan")`/`float("inf")` 不满足 \(0\le p<1\)（nan 比较恒 False，必须 isfinite）；p≥1 时 \(1/(1-p)\) 除零。
5. **p=0 恒等的「严格」**：`torch.equal` 而非 allclose；别做多余 dtype 转换或 clamp。
6. **返回新张量**：不要 in-place 乘 x。
7. **函数体内无 RNG**：mask 已给，内部再 `torch.rand` 既破坏确定性契约，也与 `dropout` 题（seed 造 mask）的分工冲突。
8. **秩校验先行**：dropout2d 收到 3 维/5 维 x 抛 `ValueError`，不要静默当 droppath 处理；mode 白名单外（如 `"DropPath"`、`"dropout1d"`）同样 `ValueError`。

## 与邻近题的区别

- **dropout**：逐元素独立 Bernoulli，mask 由 seed 经 `torch.Generator` 内部生成，判题对拍全张量；本题成块（整样本/整通道）、mask 显式入参、完全无 RNG。两题共享 inverted scaling 记账，但 mask 粒度与随机性来源都不同——「用 seed 造 mask」（dropout）与「mask 即入参」（本题）是刻意分工。
- **batchnorm-forward**：训练/推理状态的另一维度——那边用 training 开关切 batch 统计 vs running 统计；本题是训练态正则的前向本身，推理形态（p=0、全 1 mask）只是退化点，没有分支逻辑。
- **`F.dropout2d` training=False**：恒等返回、不缩放；本题始终按给定 p 缩放——「有没有丢弃」由 mask 决定、「缩不缩放」由 p 决定，两者解耦。
- **timm DropPath**：RNG 采样 mask 的训练态实现；本题把它改造成确定性算子（mask 入参），公式与 timm 的 `x.div(keep_prob)·mask` 一致。

## 对拍口径与复杂度

- **独立 oracle**：纯 python 递归/循环手工广播 + 逐元素 \(\cdot m\cdot\) scale，与张量化解法分路径——droppath 按任意秩递归走树、2d/3d 逐 \((n,c)\) 块；`allclose` 1e-6（float64 1e-12），形状与 dtype 逐一 assert。
- **判别探针**：被丢块 `torch.all(==0)`、存活块 `==x/(1-p)`；\((B,)\equiv(B,1,\dots,1)\)；bool ≡ 0/1 float；全 1 mask → \(y=x/(1-p)\)；p=0 严格恒等（`torch.equal`）；p / mask 形状 / mode 的 `ValueError` 全分类。
- **确定性数据**：黄金比例哈希伪随机张量与阈值化 bool mask（无 RNG），覆盖 p ∈ {0, 0.25, 0.5, 0.875}、秩 2/3/4/5、float32/float64。
- **复杂度**：时间 \(O(\text{numel})\)——一次广播乘（mask 元素数远小于 numel）加一次标量缩放；形状白名单校验 \(O(1)\)。空间 \(O(\text{numel})\)（新张量）。
