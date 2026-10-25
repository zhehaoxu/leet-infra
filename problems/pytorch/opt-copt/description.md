# 优化器：C-Optimizer 连续余弦门与截幅

## 题目描述

Cautious 类优化器用一道 0/1 掩码把"更新方向与梯度冲突"的坐标置零：\(g\odot u>0\) 的存活、其余清零。这个门**硬**——要么全信、要么全砍，掩码翻转时更新不连续。C-Optimizer（ByteDance Seed 技术报告思路，论文编号**待回源**）把这道门换成 \(\cos(g,m)\) 驱动的**连续插值**：梯度与动量越对齐放行越多、越对抗压制越狠，再叠一道 **sign 保持的逐元素截幅**管住最坏位移。在 AdamW 骨架上，每步是

\[
\cos=\frac{\langle g,m\rangle_F}{\|g\|_F\,\|m\|_F},\qquad
\text{gate}=\mathrm{clamp}\!\left(\frac{\cos+1}{2},0,1\right),\qquad
u=\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\varepsilon},\qquad
\theta\leftarrow\theta-\text{lr}\cdot\mathrm{clip}(u\cdot\text{gate},-\rho,\rho).
\]

门是 cautious 掩码的连续化：\(\cos=1\) 完全放行（退回 AdamW）、\(\cos=-1\) 完全压死、\(\cos=0\) 放行一半。实现它：**继承 `torch.optim.Optimizer`**，按固定梯度序列反复 `step()` 就地更新。CPU float32、全程无随机性。

## 规则

- 工厂函数接受 `params` 与 `lr`、`betas`、`eps`、`weight_decay`、`rho`，返回 `torch.optim.Optimizer` 子类实例。完整签名、参数名与默认值只写在 starter 的注释里，那是唯一口径。
- 对第 \(t\) 步、梯度 \(g\)（状态 \(m\)、\(v\) 全零起步，步计数逐参数独立、从 1 起），按**固定顺序**：
  1. **解耦权重衰减**（仅当 `weight_decay≠0`，先于一切矩估计）：\(\theta\leftarrow\theta(1-\text{lr}\cdot\lambda)\)；
  2. \(m\leftarrow\beta_1m+(1-\beta_1)g\)；\(v\leftarrow\beta_2v+(1-\beta_2)g\odot g\)；
  3. **对齐余弦**（用**本步更新后的** \(m\) 与当前 \(g\)，整张量 Frobenius 内积/范数）：\(\cos=\langle g,m\rangle_F/(\|g\|_F\|m\|_F)\)；任一范数为 0 时约定 \(\cos=1\)（不压制）；
  4. **连续插值门**：\(\text{gate}=\mathrm{clamp}((\cos+1)/2,0,1)\)；
  5. **偏差修正与更新方向**（与 `opt-adamw` 同口径）：\(u=\dfrac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}\)；
  6. **先乘门、再截幅**（顺序固定）：\(u\leftarrow u\cdot\text{gate}\)；\(u\leftarrow\mathrm{clip}(u,-\rho,\rho)\)——截幅保持符号、逐元素夹到 \([-\rho,\rho]\)；
  7. \(\theta\leftarrow\theta-\text{lr}\cdot u\)。
- 无权重衰减时每步 \(|\Delta\theta|\le\text{lr}\cdot\rho\) 恒成立（门只会把位移进一步压小）。
- `grad is None` 的参数整体跳过（连权重衰减也不施加）。
- 全程 float32、CPU、确定性；不许调用现成的 COPT/cautious 实现。

## 示例

两个参数 \(W_0=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\)、\(b_0=[1.0,-0.5,0.2]\)，`lr=0.1`、`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0.01`、`\rho=0.5`，依次喂入三步梯度（行大写为矩阵参数、行小写为一维参数）：

\(g_1: [[0.2,-0.1,0.05],[0.1,0.3,-0.2]],\ [0.1,-0.2,0.05]\)；\(g_2: [[-0.15,0.2,0.1],[0.05,-0.1,0.3]],\ [-0.05,0.1,0.2]\)；\(g_3: [[0.1,0.05,-0.2],[-0.3,0.15,0.1]],\ [0.2,-0.1,-0.1]\)。

第 1 步展开：衰减 \(W\cdot0.999\)；\(m_1=0.1g_1\) 与 \(g_1\) 严格同向 ⇒ \(\cos=1\)、\(\text{gate}=1\)；首步偏差修正后 \(u=g/(|g|+\varepsilon)\approx\pm1\)，**逐元素全被 \(\rho=0.5\) 夹住**，位移恰为 \(\pm\text{lr}\cdot\rho=\pm0.05\)。三步轨迹（\(W\) 展平）：

```
t=1  W=[0.449500, -0.149800, 0.049900, 0.249700, 0.349600, -0.049900]   b=[0.949000, -0.449500, 0.149800]
t=2  W=[0.442595, -0.176107, -0.000150, 0.199450, 0.320328, -0.067751]   b=[0.926044, -0.427043, 0.099650]
t=3  W=[0.415041, -0.213305, 0.017276, 0.230472, 0.270242, -0.099634]   b=[0.875117, -0.388986, 0.071377]
```

第 1 步每个坐标都贴在 \(\pm0.05\) 的截幅上；第 2 步起梯度转向，\(\cos<1\)、门开始缩放，位移普遍低于上界——"门在对抗方向上刹车"就是这条轨迹的形状。

## 约束

- \(\rho>0\)，\(0\le\beta_{1,2}<1\)；`params` 可混合任意形状张量（cos 按整张量算，门与截幅是标量/逐元素混合）
- 每步后的参数须与独立重算逐元素一致（`atol=rtol=1e-5`）
- 三个判别口径评测直接断言：**恒定方向**的梯度序列 ⇒ \(\cos=1\)、门恒 1、\( |u|<\rho \) 时整条轨迹与原生 AdamW 逐位一致（`atol=rtol=1e-6`）；先喂 20 步 \(+g\) 再喂 \(-0.5g\) 的对抗步 ⇒ 该步参数位移 \(<10^{-7}\)（门几乎压死）；无衰减时任意步 \(|\Delta\theta|\le\text{lr}\cdot\rho+\text{float32}\) 余量
