# 详解：Sophia-II 裁剪二阶更新

## 出处

- 算法出处：Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*，arXiv:**2305.14342**（已核实；注意某些登记材料把它写成 2305.14373，那不是本文）。本题实现的是该文 Algorithm 3 的 **Sophia-H** 变体：对角 Hessian 由 Hutchinson 型随机估计 \(\hat H= \mathrm{diag}(\partial^2\ell)\)（对角线用 \(Jv\) 型探针估计）经 EMA 收敛。判题环境离线、装不到原论文，因此**本题的算法口径完全以题面与 starter 注释为准**：分母 \(\max\{H,\varepsilon\}\) 一次方、逐元素 clip 到 \([-\rho,\rho]\)、\(\hat H\) 作显式入参。
- 库侧实现：kozistr/pytorch_optimizer 内置 `Sophia`（多为 `SophiaG`/`SophiaH` 形态，具体文件与属性命名**待回源**）。真实实现里 \(\hat H\) 由 Hutchinson 探针在 `step` 内部计算（`hessian=...` 或 Closure 返回）；本题为了 CPU 确定性把 \(\hat H\) **提为显式入参**，估算频率改由评测方的喂入序列控制——数学上等价于论文的 \(k\) 步复用。
- 一句话记忆：**Sophia = Adam 把 \(\sqrt{v}\) 换成曲率 \(H\)、把"自适应降步"换成"硬 clip"**。分母越有 curvature 信息，clip 就越只兜底而不常触发；训练早期 \(H\) 还没热起来，clip 就是事实上的主力。

## 完整推导与下标约定

设某参数第 \(t\) 步（`step()` 第 \(t\) 次调用，状态内计数从 1 起）。三个量的职责划分是理解本题的钥匙：

- **\(m_t\)（梯度 EMA，β₁）**：决定**方向**。它始终每步更新，与是否估算 Hessian 无关。
- **\(H_t\)（曲率 EMA，β₂）**：决定**步长**。它只在"估算步"更新：\(H_t=\beta_2H_{t-1}+(1-\beta_2)\hat H^t\)；非估算步 \(H_t=H_{t-1}\)（**冻结**，不是置零、也不是跳过该参数）。论文里估算每 \(k\) 步一次，本题由 `hessians is None` 编码这个开关——一个评测序列里估算步可以任意稀疏（评测里就造了"每 3 步估一次"的 20 步序列）。
- **clip（\(\rho\)）**：决定**信任域**。更新方向 \(u_t=m_t/\max\{H_t,\varepsilon\}\) 的每个分量被夹到 \([-\rho,\rho]\)。因为 \(\varepsilon\) 只兜住分母下界，**位移上界**完全由 clip 给出：无权重衰减时 \(|\Delta\theta_t|=\text{lr}\cdot|u|\le\text{lr}\cdot\rho\) 恒成立（评测用 10 步序列验证过总和上界）。

两步式 vs 一步式的写法差异：参考解写作 \(u=(m/h_{\text{clamp}}).\mathrm{clip}(-\rho,\rho)\)（除法与夹紧**分两个算子**）；独立 oracle 写成先算 \(u=m/\mathrm{clamp}(h,\varepsilon)\)、再 `clamp`，张量全部新建。两者数值上逐位一致（同样的两个浮点算子、同样的顺序），但**把除法和 clip 融合成一个复合表达式**（如 `m / h.clamp(eps).clamp(-rho, rho)` 之类）或先 clip 分母再相除都是错的——后者把限幅的职责错误地转移到分母上。

关于 \(\sqrt{H}\) vs \(H\)：预条件 \(m/H\) 对曲率的响应是**一次反比**——曲率翻倍步长减半。写成 \(m/\sqrt{H}\)（Adam 口径的肌肉记忆）会让 \(\hat H\) 翻倍时位移只乘 \(1/\sqrt2\approx0.707\)。评测有一条专项断言（\(\beta_1=\beta_2\) 固定、单一梯度步、\(\hat H_2=2\hat H_1\)、\(\rho\) 极大使 clip 不参与）：两次位移之比必须**恰为 2**。这条把"一次方"从题面文字变成可判定事实。

关于 EMA 从零起步：\(H_0=0\)，故 \(H_t=(1-\beta_2)\sum\beta_2^{t-s}\hat H^s\) 是**带亏欠系数**的平均（前几步 \(H\) 只有 \(\hat H\) 的 \((1-\beta_2)\) 倍、再乘 \(\beta_2\) 的衰减）。β₂=0.99 时第 1 步 \(H_1=0.01\hat H\)——数量级骤降、除法结果巨大、几乎全被 \(\rho\) 夹住。这不是"前几步坏了"，而是 clip 设计的用途：**在曲率估计还不可信的早期用信任域兜底**。它也解释了退化口径为什么必须取 \(\beta_2=0\)（\(\hat H\equiv1\) 直接进 \(H\)）才能对拍 SGD。

## 核心与易错点

1. **一次方，不是平方根**。题面与两条专项断言（翻倍减半、退化 SGD）都在钉它。最典型的错误是把 Adam 的 `sqrt(v)` 惯性带进来写成 `m / h.sqrt().clamp_min(eps)`。
2. **限幅在 clip，不在分母**。有人会把 \(\varepsilon\) 调大去"防爆炸"——错：\(\varepsilon\) 只改小分母的下界（\(H=0\) 时除法结果由 \(m/\varepsilon\) 给出，通常更大），真正的 \(|\Delta\theta|\le\text{lr}\cdot\rho\) 由 clip 保证。评测专门造了 \(\hat H\equiv0\) 的步：位移必须**恰好** \(\text{lr}\cdot\rho\)，且方向逆着 \(m\) 的符号。
3. **`hessians` 的对齐与占位**。`hessians` 与**全部参数**（跨 group 拉平）同长同序；`grad is None` 的参数位置照样要给一个张量（评测给全零占位）。按 group 内参数直接 `zip` 而不查全局顺序的实现，在多 group 下会静默错位。参考解用 `{id(p): i}` 建全局下标表。
4. **\(H\) 冻结 ≠ 跳过参数**。`hessians=None` 的步只冻结 \(H\)：\(m\) 照常 EMA、\(\theta\) 照常走 \(-\text{lr}\cdot\mathrm{clip}(m/\max\{H,\varepsilon\},\rho)\)。把它实现成"没有 \(\hat H\) 就整步跳过"会让参数在非估算步完全停走——与论文的 \(k\) 步复用语义相反，评测用"估一次后连走两步纯梯度步仍持续下降"的断言打它。
5. **解耦权重衰减在最前**：\(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\) 先做，\(m\) 用**原始** \(g\)（与 opt-adamw 同口径）。衰减不参与 \(m\)、\(H\) 的统计。
6. **状态全零起步、EMA 不做偏差修正**。Sophia 不像 Adam 除 \(1-\beta^t\)：\(H\) 的亏欠（偏小）由 clip 兜底，\(m\) 的亏欠在早期同样偏小——这是算法本身的取舍，别"顺手"加偏差修正，评测轨迹会立刻对不上。
7. **就地更新**：全部通过 `p.mul_/add_/addcdiv_` 或 `copy_` 写回原张量，返回值只是 `closure` 的透传。新建张量再赋值 `p = ...` 会静默丢失（参数列表仍持旧对象）。
8. **float32 与容差**：逐元素比对用 `allclose(atol=1e-5, rtol=1e-5)`。clip 的边界（恰等于 \(\pm\rho\)）是浮点敏感区——先除后夹、不要把两次夹紧写成一次 `clamp(h, eps, 1/ρ)` 之类。

## 与邻近题的区别

- **vs `opt-adamw`**：骨架同（解耦衰减、双 EMA、偏差修正……），两处分叉：① 分母 \(\sqrt{v}\) ← 由 \(g^2\) 自算，Sophia 是 \(\max\{H,\varepsilon\}\) ← 曲率估计一次方且由外部喂入；② Adam 无 clip（\(\sqrt{v}\) 天然限步），Sophia 有逐元素 \(\pm\rho\) clip。所以 Adam 侧"翻倍梯度→位移翻倍"的直觉在 Sophia 侧被 clip 打破。另注意 Sophia **不做** \(1-\beta^t\) 偏差修正。
- **vs `opt-copt`（C-Optimizer）**：两题都带 \(\pm\rho\) 截幅、都继承 Optimizer，但管的东西不同——copt 的 clip 作用在**Adam 式更新方向** \(u\) 上，且 clip 之前还有一道 cos 门（连续缩放）；Sophia 的 clip 作用在**曲率预条件**的 \(m/H\) 上，没有方向门。所以 copt 的 \(|\Delta\theta|\le\text{lr}\cdot\rho\) 会被门先缩、常常达不到上界，而 Sophia 早期几乎步步贴上界——用"位移是否常等于 lr·ρ"就能把两题的轨迹区分开。
- **vs `opt-galore`（GaLore）**：两者都消费"额外的二阶/结构信息入参"，但 GaLore 的 \(P\) 是**梯度所在子空间的正交基**（投影后跑 Adam 再映回），题面喂入的是矩阵因子；Sophia 的 \(\hat H\) 是**逐元素曲率**（逐对角元预条件，不换子空间）。前者改变更新所在的子空间，后者改变逐坐标的步长。
- **vs `opt-lookahead`**：lookahead 的"冻结/回滚"是权重层面的周期性行为，Sophia 的"H 冻结"是**状态层**面按估算频率的行为，两者正交、可叠加；前者每 k 步写 \(\theta\)，后者每 k 步写 \(H\)。

## 对拍口径与复杂度

- 主口径：独立重算 \(\_oracle\_sophia\)——不用 Optimizer 类，纯 torch 逐元素按 Algorithm 3 写（m/H 用新张量、除法与夹紧分两步），与参考解的算子顺序刻意错开；估算步稀疏（每 3 步一次）的 20 步长序列逐步对拍，`allclose(atol=1e-5, rtol=1e-5)`。
- 硬参照：\(\beta_1=\beta_2=0\)、\(\hat H\equiv1\)、\(\rho=10^{12}\)、wd=0 时与**原生 `torch.optim.SGD`** 逐位一致（\(10^{-7}\)）。
- 结构断言：\(\hat H\) 翻倍 ⇒ 位移比恰 2（不是 \(2^{-1/2}\)）；\(\hat H\equiv0\) ⇒ 位移恰 \(\text{lr}\cdot\rho\)、方向逆 m；无衰减下 10 步总位移 ≤ 10·lr·ρ（float32 留 1e-5 余量）；`grad is None` 参数逐位不动。
- 复杂度：每参数每步 \(O(\text{numel})\)（一次 EMA、一次除法、一次 clamp）；估算步额外一次 \(\hat H\) EMA，同阶。额外显存一份 \(H\)（与参数等大），比 Adam 的 \(v\) 同价。
