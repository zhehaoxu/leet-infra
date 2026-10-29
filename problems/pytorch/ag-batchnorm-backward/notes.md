# 详解：BatchNorm 反向的三段式与 1/N 账

## 出处

BatchNorm 反向是「卷积网络训练框架」岗位的经典手写题，也是论文 *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* 附录里那三个公式的复现。面试的区分点从来不是能否背公式，而是能否说清 \(1/N\) 从哪来、以及为什么 `N=1` 时输出必然是 0。

## 三段式的来源与 1/N 因子

对固定特征列，记 \(a_n=dY_n\gamma_n\)，\(\hat x_n=(x_n-\mu)/\sigma\)，\(\mu=\frac1N\sum_k x_k\)，\(\sigma=(\frac1N\sum_k(x_k-\mu)^2+eps)^{1/2}\)。则

$$\frac{\partial L}{\partial x_j}=\frac{a_j}{\sigma}-\frac{1}{\sigma^3}\Big(\frac1N\sum_k a_k\Big)(x_j-\mu)-\frac{1}{\sigma^3}(x_j-\mu)\frac1N\sum_k a_k\hat x_k$$

第一项直通；第二项来自 \(\partial\mu/\partial x_j=1/N\)；第三项来自 \(\partial\sigma/\partial x_j=(x_j-\mu)/(N\sigma)\)。**两个非直通项各带一个 \(1/N\)**，写成 \(\text{mean}_N(\cdot)\) 就自动带上了；如果有人把求和与均值混用（`sum` 少除一次 N），就是「双重计入」的镜像错误——系数会差 N 倍，在 `eps=0` 自检 \(\sum_N dX=0\) 上立刻暴露。

## 核心与易错点

- **训练态与推理态不同**：推理用 `running_mean/running_var`，它们对 `x` 不可导，\(dX=dY\gamma/\sigma_{run}\) 只有一段；本题考训练态。
- **N=1**：\(\mu=x\)、\(var=0\)，`eps>0` 时三段相消给 0，测试用这条钉死公式的自洽性。
- **dGamma/dBeta 沿 batch 求和（不取均值）**：形状 `(C,)`，与 LayerNorm 的「沿前导轴折叠」方向不同。
- **gamma 全 1、beta 常数不影响 dX**：`beta` 完全不进 dX；改 `beta` 后三个返回必须逐位不变。
- **有偏方差**：PyTorch 的 BN 用 \(1/N\) 而非 \(1/(N-1)\)。

## 与邻近题的区别

`ag-layernorm-backward`/`ag-rmsnorm-backward` 沿特征末轴归一、每个样本独立，因此不产生跨样本耦合；本题沿 batch 轴归一，一个样本的扰动会传到整列，这正是三段式的成因。`batchnorm-forward` 类题只求正向与 running 统计量更新。

## 对拍口径与复杂度

oracle 用 `F.batch_norm(x, None, None, gamma, beta, training=True, eps=eps)` 加 `torch.autograd.grad` 取三个梯度，float64 atol=1e-9。示例取 `eps=0` 且 \(x\) 为 \(\pm1\)，使 \(\mu=0,\sigma=1\)，四项均可手算。复杂度 \(O(NC)\)，两次沿 batch 的归约 + 一次逐列广播。
