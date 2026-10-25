# 详解：Prodigy 自适应学习率（D-Adaptation）

## 出处
Defazio & Mishchenko, *Learning-Rate-Free Learning by D-Adaptation*（arXiv:2301.07733），以及其 Adam 实用化实现 Prodigy。参考实现取 kozistr/pytorch-optimizer 的 `Prodigy`（默认 `lr=1.0`、`betas=(0.9,0.999)`、`beta3=√beta2`、`d0=1e-6`、`d_coef=1.0`、`growth_rate=inf`、`bias_correction=False`、`safeguard_warmup=False`、`eps=1e-8`、`weight_decouple=True`）。CPU 上 `foreach` 不触发，走逐参数路径，本详解即该路径。

## 核心思想：在线估计“到最优点的距离 D”
Adam 的步长 \(lr\) 需要人调。D-Adaptation 观察到：若已知参数初值到最优解的距离 \(D\)，则最优步长可由 \(D\) 推出。Prodigy 用一层“对偶式”估计在线逼近 \(D\)：

- \(\mathit{num}\)（组级标量，跨步 EMA）累积每步的 \( \langle g,\ p_0-p\rangle \)——即梯度与“相对初位移”的内积，正比于函数值下降量；
- \(\mathit{den}\)（每步重算）是 \(\sum|s|\)，其中 \(s\) 是梯度的 \(\beta_3\)-EMA，正比于步长的“代价”；
- 距离估计 \(\hat d = d_{coef}\cdot \mathit{num}/\mathit{den}\)；再把 \(d\) 乘进等效步长 \(d_{lr}=d\cdot lr\)。

于是 \(lr\) 保持 1 即可，\(d\) 自己长大到合适尺度。

## 为什么难写对（易错点）
1. **两趟遍历**：先对**所有**参数做统计（更新 \(m,v,s,\mathit{num},\mathit{den}\)），**再**统一算 \(d\)、**然后**第二趟才真正更新 \(p\)。单趟写法会错。
2. **旧 \(d_{lr}\) vs 新 \(d\)**：第二趟的步长 `value=-d_lr` 用的是**步首旧 \(d\)** 算出的 \(d_{lr}\)，而分母 `√v + d·eps` 用的是**刚更新后的新 \(d\)**。
3. **`num` 先衰减再累加**：每步开头 \(\mathit{num}\leftarrow\beta_3\mathit{num}\)，随后才 `+=`；而 \(\mathit{den}\) 每步清零重算。
4. **\(d\) 单调、带 growth_rate 上限**：首步 \(d==d_0\) 才 `max(d0, d̂)`；此后 \(d_{\max}=\max(d_{\max},\hat d)\)、\(d=\min(d_{\max}, d\cdot\)growth_rate\()\)。`growth_rate=inf` 时 \(d=d_{\max}\)（单调不减）。
5. **矩里也带 \(d\)**：\(m\leftarrow\beta_1 m + d(1-\beta_1)g\)、\(v\leftarrow\beta_2 v + d^2(1-\beta_2)g^2\)（注意 \(d\)、\(d^2\) 系数，与普通 Adam 不同）。

## 对拍口径
判题用独立重算 `_oracle_prodigy`（把两趟写成 list 版、显式 `num/den` 标量），与参考解在多形状、多步随机梯度下 `allclose(atol=1e-5, rtol=1e-5)` 比对；本合集实测逐元素**完全相同**（maxdiff=0），因为两者都走同一 CPU 标量路径。`check_none_grad_param_skipped` 验证无梯度参数两趟都跳过、连 \(p_0/s\) 都不建。

## 复杂度
每步两趟 \(O(\text{numel})\)；额外状态 \(m,v,s,p_0\)（约四份，比 Adam 多一份 \(s\) 与快照 \(p_0\)）。
