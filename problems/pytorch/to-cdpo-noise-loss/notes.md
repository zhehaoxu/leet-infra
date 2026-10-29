# 解题说明 · cDPO

## 噪声模型：对称翻转

cDPO 假设每条偏好标签以**相同**概率 $\varepsilon$ 被翻成相反标签，且翻转与否与样本本身无关（对称噪声）。在这个模型下，观测到「chosen 胜出」的概率是 $(1-\varepsilon)\sigma(x)+\varepsilon(1-\sigma(x))$，取其负对数即得软标签交叉熵。$\varepsilon$ 因此有非常直接的语义：**你认为有多大比例的数据是错的**。

## 为什么 ε 封顶 0.5

$\varepsilon>0.5$ 意味着「观测标签比随机猜测还不可信」，此时更合理的做法是把标签整体反置，而不是继续平滑；数学上 $\varepsilon$ 与 $1-\varepsilon$ 在损失里对称交换，$\varepsilon>0.5$ 等价于先翻转数据再用 $1-\varepsilon$。所以契约把范围限定在 $[0,0.5]$，$\varepsilon=0.5$ 是「标签不含任何信息」的极端：软标签对称，损失在 $x=0$ 处取最小值 $\ln2$，模型只会被推向「无法区分」，不会偏向任何一侧。

## 稳定性与两个退化点

两个 log-sigmoid 中的 $\log\sigma(-x)$ 在 $x$ 很大时会有 $\sigma(-x)\to 0$，直接 `log(sigmoid())` 会下溢成 $-\infty$。稳定式 $\log\sigma(z)=\min(0,z)-\log(1+e^{-|z|})$ 对任意 $z$ 都安全。两个必须钉住的退化点：$\varepsilon=0$ 与 DPO 逐位一致；$x=0$ 时 $\log\sigma(0)=\log\sigma(0)=-\ln2$，加权后仍是 $-\ln2$，与 $\varepsilon$ 无关——测试对 $\varepsilon\in\{0,0.1,0.25,0.5\}$ 全查。

## 与鲁棒 DPO 理论的关系

Ray Chowdhury et al. 证明：在对称标签噪声下，带平滑的 DPO 目标的最优解与干净数据下的最优解之间只差一个有界偏差，偏差随 $\varepsilon$ 增大；这与「平滑只是给已分对样本加反向拉力」的直觉一致。cDPO 因此是一个**保守**的修正：它不尝试识别哪些标签错了，只用统一的软标签限制 logit 的绝对尺度。
