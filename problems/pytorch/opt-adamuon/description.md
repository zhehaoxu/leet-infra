# 优化器：AdaMuon 逐元素自适应正交更新

## 题目描述

Muon 把"梯度 → 动量 → Newton-Schulz 正交化"的正交更新搬进了大模型训练：更新方向在矩阵谱上被拉平，各奇异方向步长一致。但它对所有坐标一视同仁——没有 Adam 那种逐坐标的自适应步长。AdaMuon（Si et al., arXiv:**2507.11005**）在 Muon 之上叠加**逐元素 RMS 预条件**，再用**范数还原**把 Muon 的整体更新尺度原样保住：

\[
m\leftarrow\mu m+(1-\mu)g,\quad G=(1-\mu)g+\mu m,\quad O=\mathrm{NS}(G),\quad
v\leftarrow\beta_2v+(1-\beta_2)O^{\odot2},\quad
\theta\leftarrow\theta-\text{lr}\cdot U,
\]
\[
U=\frac{O}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}\cdot\frac{\|O\|_F}{\|O/(\sqrt{v/(1-\beta_2^t)}+\varepsilon)\|_F}.
\]

两个关键设计：**二阶矩的驱动信号是正交化后的 \(O\)（不是 \(g\)）**——自适应发生在 Muon 之后；**RMS 相除之后立即做范数还原**，使 \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\) 恒成立——Muon 的谱尺度不被逐坐标除法破坏。1-D 参数（偏置、归一化增益）走纯 AdamW 路径。实现它：**继承 `torch.optim.Optimizer`**，按固定梯度序列反复 `step()` 就地更新。CPU float32、全程无随机性。

## 规则

- 工厂函数接受 `params` 与 `lr`、`momentum`、`ns_steps`、`betas`、`eps`、`weight_decay`，返回 `torch.optim.Optimizer` 子类实例。完整签名、参数名与默认值只写在 starter 的注释里，那是唯一口径。
- 状态全零起步（\(m\)、\(v\)）；步计数逐参数独立、从 1 起。
- **解耦权重衰减**（仅当 `weight_decay≠0`，先于一切更新）：\(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\)。
- **2-D 参数（dim ≥ 2）**按**固定顺序**：
  1. 动量：\(m\leftarrow\mu m+(1-\mu)g\)；
  2. Nesterov：\(G\leftarrow(1-\mu)g+\mu m\)（用更新后的 \(m\)）；
  3. Newton-Schulz 正交化 \(O=\mathrm{NS}(G)\)，quintic 系数 \((3.4445,-4.7750,2.0315)\)、迭代 `ns_steps` 次：若 rows>cols 先转置；\(X\leftarrow X/(\|X\|_F+10^{-7})\)；循环 \(\{A=XX^{\!\top};\ X=aX+(bA+cA^2)X\}\)；转置过再转置回来。**不做** Muon 的 \(\max(1,\text{rows}/\text{cols})^{0.5}\) 谱缩放（范数还原会抵消它，叠加是双重缩放）；
  4. 逐元素 RMS 二阶矩：\(v\leftarrow\beta_2v+(1-\beta_2)O\odot O\)，\(\hat V=v/(1-\beta_2^t)\)；
  5. **自适应预条件 + 范数还原**（顺序固定）：\(U\leftarrow O/(\sqrt{\hat V}+\varepsilon)\)；\(U\leftarrow U\cdot\big(\|O\|_F/(\|U\|_F+10^{-12})\big)\)；
  6. \(\theta\leftarrow\theta-\text{lr}\cdot U\)。
- **1-D 参数（dim < 2）走纯 AdamW 路径**：\(m\leftarrow\beta_1m+(1-\beta_1)g\)、\(v\leftarrow\beta_2v+(1-\beta_2)g^{\odot2}\)、\(\theta\leftarrow\theta-\text{lr}\cdot\dfrac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}\)（`momentum` 只用于 2-D 动量）。
- `grad is None` 的参数整体跳过。无权重衰减时 \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\) 恒成立。
- 全程 float32、CPU、确定性；不许调用现成的 Muon/AdaMuon/AdamW 组合器（NS 用手写循环实现）。

## 示例

两个参数 \(W_0=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\)（2-D，走 Muon+Ada 路径）、\(b_0=[1.0,-0.5,0.2]\)（1-D，走 AdamW 路径），`lr=0.1`、`momentum=0.95`、`ns_steps=5`、`betas=(0.9,0.95)`、`eps=1e-8`、`weight_decay=0.01`，喂入三步梯度（同 opt-mars 示例）。

第 1 步：\(b\) 侧 AdamW 首步给出 \(u\approx\mathrm{sign}(g)\)，位移 \(\pm0.1\)（\(b_1=[0.899,-0.3995,0.0998]\)）。\(W\) 侧首步 \(\hat V=O^{\odot2}\) ⇒ \(U\approx\mathrm{sign}(O)\) 逐元素等幅，再经范数还原——**首步位移全部相等**：

```
t=1  W=[0.451636, -0.151936, 0.052036, 0.251836, 0.351736, -0.052036]   b=[0.899000, -0.399500, 0.099800]
t=2  W=[0.489015, -0.213749, -0.006608, 0.193297, 0.364099, -0.106108]   b=[0.871264, -0.372264, 0.012227]
t=3  W=[0.447945, -0.274676, 0.071185, 0.238998, 0.308142, -0.126250]   b=[0.805518, -0.326431, -0.021284]
```

第 1 步 \(W\) 的六个位移都是 \(\pm0.047864\)（sign 板），这是"首步 \(\sqrt{\hat V}=|O|\)"的直接推论；第 2 步起 \(v\) 积累出逐坐标差异，位移不再等幅，但整步的 \(\|\Delta W\|_F\) 仍被范数还原钉在 \(\text{lr}\cdot\|O\|_F\)。

## 约束

- \(\mu\in[0,1)\)、\(0\le\beta_{1,2}<1\)、`ns_steps≥1`；`params` 可混合 1-D 与 ≥2-D 张量
- 每步后的参数须与独立重算逐元素一致（`atol=rtol=1e-5`）
- 三个判别口径评测直接断言：1-D 参数与**原生 `torch.optim.AdamW`** 逐位一致（`atol=rtol=1e-6`）；首步 2-D 位移逐元素等幅（sign 板，\(\varepsilon\) 取 \(10^{-12}\) 时余量内）；无衰减下任意步 \(\|\Delta\theta\|_F=\text{lr}\cdot\|O\|_F\)（\(O\) 用测试内**独立实现的 NS** 复算）
