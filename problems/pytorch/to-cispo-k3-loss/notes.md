# 解题说明 · CISPO 与 K3

## 把裁剪从目标里挪到权重上

PPO 的 $\min(\rho A,\mathrm{clip}(\rho)A)$ 有一个常被忽视的后果：当 $\rho$ 越界、`min` 选中常数项时，该 token 对参数的梯度**变成严格的 0**。在长序列 RL 里，越界 token 占比一旦升高，就等于有相当一部分 token 白采样。CISPO 把裁剪结果当成「每个 token 的重要性权重」，权重本身有界，`logπ` 仍对所有 token 求导——于是梯度不会消失，只是被限幅。这与「信任域」的初衷不冲突：限幅的正是单 token 对梯度的贡献。

## 为什么必须停梯度

CISPO 的权重 $w=\mathrm{clamp}(\rho)$ 依赖 $\rho$，而 $\rho$ 依赖 $\log\pi$。如果不做 `detach`，对 $\log\pi$ 求导会多出 $\partial w/\partial\log\pi=\rho$ 这一项（未裁剪区间内），目标函数就变成了「带权的 logπ 加上权重自身的变化」，与论文定义的对象不同。停梯度后，$\partial L/\partial\log\pi_t=-w_tA/N$ 干净、逐 token 可解析核对，测试正是用 autograd 与这条解析式对拍。

## K3 为何非负且方向敏感

设 $r=\pi_\text{ref}/\pi$，则 $k_3=r-\log r-1$。令 $f(r)=r-\log r-1$，$f'(r)=1-1/r$，在 $r=1$ 取最小值 $f(1)=0$，所以任意 $r>0$ 都有 $k_3\ge0$。它的期望是 $\mathrm{KL}(\pi\|\pi_\text{ref})$：因为 $\mathbb{E}_\pi[r]=1$、$\mathbb{E}_\pi[\log r]=\mathrm{KL}(\pi\|\pi_\text{ref})$。方向不能写反——样本来自**当前**策略，所以比值必须是 $\pi_\text{ref}/\pi$；写成 $\pi/\pi_\text{ref}$ 会得到另一个非负量，估计的是反方向的 KL，数值完全不同。测试用一条不对称样本把两种方向区分开。

## 溢出保护为什么卡在 20

$k_3$ 里的 $e^{\text{log\_r}}$ 在 $\text{log\_r}$ 很大时溢出。float32 的指数上限约 88，float64 约 709，但 $k_3$ 的数值在 $\text{log\_r}>20$ 时已经大到没有训练意义（$e^{20}\approx4.85\times10^8$），而且真实 logprob 差达到这个量级通常意味着数据或掩码出了问题。契约因此在 20 处直接报错，把「静默地算出 inf 再污染整个 batch」变成显式的输入错误。

## 裁剪比例作为诊断量

`clip_frac` 统计越界 token 的比例，是判断「策略漂移多大 / 裁剪半径是否合适」的廉价指标。它用**严格**不等号：$\rho$ 恰等于 $1+\epsilon_\text{high}$ 时不算越界，因为此时 $\mathrm{clamp}$ 不改变数值、也不会触发任何行为差异。这个小约定让「边界」与「越界」在统计上互斥，避免实现用 `>=` 时把边界 token 误计。
