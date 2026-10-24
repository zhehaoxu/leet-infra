# 解题说明 · Bradley-Terry 奖励模型

## 池化选"最后有效 token"

奖励是整条回答的标量，但骨干是逐 token 输出的。取最后有效 token 的隐藏态作序列表示，是因为因果模型里最后一个真实 token 已聚合了全序列信息（类似用 end-of-sequence 位置做分类）。掩码不保证连续（本题用 `(mask*arange).max` 直接取最大有效下标），比"假设 padding 只在右端"更稳健。

## Bradley-Terry 与 DPO 的关系

本题损失形如 $-\log\sigma(r_w - r_l)$，与 DPO 只差在 $r$ 是显式奖励模型输出、而 DPO 用 $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ 当隐式奖励——两者其实是同一 logistic 骨架。训练得到的 $r_\phi$ 正是下游 PPO 里给回答打分的模型。

## 为何用稳定 log-sigmoid

reward 差可达较大正值/负值，`log(sigmoid(x))` 在 $x$ 很负时下溢。统一用 `F.logsigmoid`（$\min(0,x)-\log(1+e^{-|x|})$），oracle 用 `math.log1p` 独立复刻，避免判题假绿。

## 分数无界的注意点

价值头输出的是未过 sigmoid 的裸标量 reward，可正可负、无范围；只有"chosen 与 rejected 之差"进入损失，因此给每条加常数偏置不影响 loss（差消掉），bias 主要影响下游 PPO 的基线而非排序。
