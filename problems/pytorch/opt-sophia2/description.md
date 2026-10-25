# 优化器：Sophia-II 裁剪二阶更新

## 题目描述

Adam 家族用梯度的二阶矩 \(\sqrt{v}\) 当分母，二阶信息永远只是"梯度大小的滑动平均"。Sophia（Liu et al., arXiv:**2305.14342**）把分母换成**对角 Hessian 估计的一次方**：预条件量由"梯度烈度"升级为"曲率"，再用一道**逐元素 clip** 承担限幅，避免 Hessian 估计方差大时步子爆掉。论文的 Algorithm 3（Sophia-H 变体）每步更新为

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
H_t=\beta_2 H_{t-k}+(1-\beta_2)\hat H^t,\qquad
\theta_t=\theta_{t-1}(1-\text{lr}\cdot\lambda)-\text{lr}\cdot\mathrm{clip}\!\left(\frac{m_t}{\max\{H_t,\varepsilon\}},-\rho,\rho\right),
\]

其中 \(\hat H\) 是 Hutchinson 型随机估计——真实训练里每 \(k\) 步才算一次（贵），中间步**冻结 \(H\)** 照常用旧值。本题把 \(\hat H\) 做成**显式入参**：评测按固定序列把 (梯度, Hessian 估计) 一起喂进来，CPU 上就能精确判定。

实现它：**继承 `torch.optim.Optimizer`**，`step(closure=None, hessians=None)` 就地更新 `params`。CPU float32、全程无随机性。

## 规则

- 工厂函数接受 `params` 与 `lr`、`betas`、`eps`、`weight_decay`、`rho`，返回 `torch.optim.Optimizer` 子类实例。完整签名、参数名与默认值只写在 starter 的注释里，那是唯一口径。
- `step(closure=None, hessians=None)`：
  - `hessians` 与参数**同长同序**（跨所有 param group 拉平后的顺序）；无梯度的参数位置也要占位（评测会喂全长的 `hessians`）。`hessians=None` 表示**本步不做 Hessian 估算**：\(H\) 冻结不动，但 \(m\) 与 \(\theta\) 照常更新——这就是论文 \(k\) 步复用语义的确定性表达。
  - `closure` 非空时在 `enable_grad` 下调用并把返回值原样返回。
- 每步（对每个 `grad` 非 `None` 的参数）按**固定顺序**：
  1. **解耦权重衰减**（仅当 `weight_decay≠0`，先于一切矩估计）：\(\theta \leftarrow \theta\,(1-\text{lr}\cdot\lambda)\)；
  2. \(m \leftarrow \beta_1 m+(1-\beta_1)g\)；
  3. 仅当本步给了 `hessians`：\(H \leftarrow \beta_2 H+(1-\beta_2)\hat H\)（EMA 从全零状态起步）；
  4. \(u \leftarrow m \,/\, \max\{H,\varepsilon\}\)（分母逐元素 `clamp_min(ε)`），再 \(\mathrm{clip}(u,-\rho,\rho)\)；
  5. \(\theta \leftarrow \theta - \text{lr}\cdot u\)。
- **分母是 \(H\) 的一次方** `max{H, ε}`，不是 \(\sqrt{H}\)（\(\sqrt{\cdot}\) 是 Adam 的口径）。推论：\(\hat H\) 整体翻倍 ⇒ 未夹紧时的位移**精确减半**。
- **限幅只由 clip 承担**：逐元素夹到 \([-\rho,\rho]\)。无权重衰减时每步 \(|\Delta\theta| \le \text{lr}\cdot\rho\) 恒成立。
- 状态初始化全零（`exp_avg`、`hessian`、`step`）。\(H\) 也从 0 做 EMA：β₂=0.99 时前几步 \(H\) 只有 0.01 量级、分母主要靠 \(\varepsilon\) 兜底，更新几乎都被 \(\rho\) 夹住——这是预期行为，不是 bug。
- `grad is None` 的参数整体跳过（不衰减、不动 \(m/H\)、不写 \(\theta\)）。
- 全程 float32、CPU、确定性；不许调用现成的 Sophia 实现。

## 示例

单参数 \(\theta_0=[1.0,\,-2.0,\,0.5]\)，`lr=0.1`、`betas=(0.9,0.99)`、`eps=1e-8`、`weight_decay=0.1`、`\rho=0.5`，依次喂入三步 (梯度, Ĥ)：

\(g_1=[0.10,-0.20,0.30]\)、\(\hat H_1=[0.5,1.0,0.25]\)；\(g_2=[-0.40,0.05,0.20]\)、\(\hat H_2=[0.5,0.5,0.5]\)；\(g_3=[0.15,-0.10,-0.25]\)、\(\hat H_3=[2.0,1.0,0.1]\)。

第 1 步展开：权重衰减 \(\theta\cdot0.99=[0.99,-1.98,0.495]\)；\(m_1=0.1g_1=[0.01,-0.02,0.03]\)；\(H_1=0.01\hat H_1=[0.005,0.01,0.0025]\)；\(m/\max\{H,\varepsilon\}=[2,-2,12]\) —— 三个分量**全部被 \(\rho=0.5\) 夹住**，落点是 \(\theta_1=[0.94,\,-1.93,\,0.445]\)。三步轨迹：

```
t=1  θ=[0.940000, -1.930000, 0.445000]
t=2  θ=[0.980600, -1.860700, 0.390550]
t=3  θ=[1.014009, -1.792093, 0.336644]
```

第 1 步三个方向的更新都被夹紧（EMA 从零起步 ⇒ 早期分母极小），这正是"前几步靠 \(\rho\) 限幅"的活样本；到第 3 步 \(H_3=[0.0247,0.0099,0.0026]\) 量级仍小，位移依旧受 clip 约束。

## 约束

- \(\rho>0\)，\(0\le\beta_{1,2}<1\)；`params` 可混合任意形状的张量（分母与 clip 都是逐元素的）
- 每步后的参数须与独立重算逐元素一致（`atol=rtol=1e-5`；退化口径见下）
- 三个判别口径评测直接断言：\(\hat H\) 整体翻倍 ⇒ 位移精确减半（写成 \(\sqrt{H}\) 会得 \(1/\sqrt2\)）；\(H\approx0\) 时位移恰为 \(\text{lr}\cdot\rho\) 且方向逆着 \(m\) 的符号，且任意步 \(|\Delta\theta|\le\text{lr}\cdot\rho\)；\(\beta_1=\beta_2=0\)、\(\hat H\equiv1\)、\(\rho\) 极大、无衰减时逐位退化为原生 SGD（必须取 \(\beta_2=0\)：EMA 从 0 起步，\(\beta_2=0.99\) 时前几步 \(H\) 只有 0.01 量级，退化不成立）
- `hessians=None` 的步：\(H\) 逐位不变，但 \(m\)、\(\theta\) 照常更新
