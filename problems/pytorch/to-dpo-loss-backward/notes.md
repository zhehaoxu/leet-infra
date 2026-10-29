# 解题说明 · DPO 解析反向

## 三步链式法则

令 $x=\beta\Delta$，$L_b=-\log\sigma(x_b)=\mathrm{softplus}(-x_b)$。对 $\log\sigma$ 求导是 $\frac{d}{dx}\log\sigma(x)=\sigma(-x)$，因此

$$\frac{\partial L_b}{\partial x_b}=-\sigma(-x_b).$$

再穿过 $x=\beta\Delta$ 与 $\Delta=(s_w-\text{ref}_w)-(s_l-\text{ref}_l)$ 两层线性变换：

$$\frac{\partial L_b}{\partial\Delta_b}=-\beta\,\sigma(-x_b),\qquad
\frac{\partial L_b}{\partial s_w}=\frac{\partial L_b}{\partial\Delta_b},\qquad
\frac{\partial L_b}{\partial s_l}=-\frac{\partial L_b}{\partial\Delta_b}.$$

最后 $s_w=\sum_t\text{mask}\cdot\text{logp}$ 把序列级梯度摊回逐 token，并除以 $B$（前向是 batch 平均）。整条链没有非线性耦合，所以闭式解非常短——但把 $1/B$、$\beta$、两侧符号、掩码任何一处漏掉，梯度就会整体错一个因子或一个方向。

## 为什么两侧符号相反

$\Delta$ 对 chosen 的 logprob 是 $+1$、对 rejected 是 $-1$。$x$ 越大损失越小，所以「提高 chosen 的 logprob」降低损失（负梯度方向），「提高 rejected 的 logprob」升高损失（正梯度）。在同一个 $\Delta$ 下，两侧逐 token 梯度大小相同、符号相反；这正好对应 DPO 的语义——把两条回答推开的不是某个绝对量，而是它们**差**的变化。

## 掩码位置必须严格为 0

padding token 的 logprob 不参与序列求和，因此对损失没有任何影响，梯度必须是精确的 0。常见错误是「先算整行梯度再乘 mask」，这在数学上正确；但若误用 `mask.float()` 之外的广播、或对 padding 位置也写入 `dL/dΔ`（比如忘了乘 mask），这些位置的梯度就会是非零常数，在优化器里变成对 padding 的伪更新。测试既检查 `grad[masked] == 0`，也检查「改动 masked 位置的 logp 不改变损失与其它梯度」。

## 均值缩放与批大小

前向是 $\frac1B\sum_b$，所以每条样本的 $\partial L/\partial\Delta_b$ 都带 $1/B$。把同一条样本复制成 batch 两个样本，损失不变但每个 token 的梯度减半——这正是测试里 `loss_scaling` 检查的内容。写自定义 kernel 时最容易忘记这个因子，因为它在前向里是「除以 B」、在反向里却要乘进每个样本。

## 数值稳定性

$\sigma(-x)$ 对很大的正 $x$ 会趋近 0，对很大的负 $x$ 趋近 1；用 `sigmoid(-x)` 直接算即可，不要写成 $1-\sigma(x)$（大 $x$ 时会因相减丢精度）。前向用 `softplus(-x)` 而不是 `log(sigmoid(x))`，后者在 $x$ 很负时下溢为 $-\infty$。这两点是 DPO 数值实现里反复出现的坑，也是本题允许用 autograd 对拍但要求解析式的现实原因：解析式可以直接嵌进融合核，且与框架梯度互为独立校验。
