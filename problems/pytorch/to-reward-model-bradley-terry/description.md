# Reward Modeling：Bradley–Terry 成对排序损失

## 题目描述

RLHF 的第一步是训练一个**奖励模型** $r_\phi(x,y)$：给定 prompt 与回答，输出一个标量分数，使人类更偏好的回答分数更高。常用 **Bradley–Terry** 模型把"成对比较"转成概率：$P(y_w \succ y_l) = \sigma(r_w - r_l)$，再用极大似然（＝对数损失）训练：

$$\mathcal{L}_\text{RM} = -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]$$

奖励模型本体通常是一个骨干网络 + 一个把隐藏态映射到标量的**价值头**（线性 $D\to1$）。HF `RewardTrainer` 的做法是：取每条序列**最后一个有效 token** 的隐藏态过价值头得到整条回答的 reward（序列池化）。本题给定 chosen / rejected 两批的逐 token 隐藏态与注意力掩码，让你完成"末位有效 token 池化 → 价值头打分 → Bradley–Terry 损失"这条链路。

## 规则

给定（batch 大小 $N$、序列长 $L$、隐维 $D$）：`chosen_hidden`、`rejected_hidden` 各 $(N, L, D)$；`chosen_mask`、`rejected_mask` 各 $(N, L)$（1=有效 token，0=padding）；`weight` $(D,)$、标量 `bias`（价值头参数）：

1. **池化**：对每条序列取**最后一个有效 token** 的隐藏态——即满足 mask$=1$ 的最大下标 $j^\ast$，取 $h = $ hidden$[b, j^\ast, :]$（$(D,)$）
2. **打分**：$r = h \cdot $`weight` $+$`bias`（点积到标量），分别得 `r_chosen`、`r_rejected`（各 $(N,)$）
3. **损失**：逐对 $-\log\sigma(r_\text{chosen} - r_\text{rejected})$，用稳定 log-sigmoid，再对 batch 取**均值**

返回 0 维标量 `torch.Tensor`＝ 平均 reward modeling loss。

## 示例

```
N=2, L=3, D=2, weight=[1.0,0.5], bias=0.1
chosen_hidden   = [[[0.1,0.3],[0.2,-0.1],[0.5,0.2]],
                   [[0.3,0.0],[0.9,0.4],[0.0,0.0]]]
rejected_hidden = [[[0.4,0.1],[0.2,0.5],[0.1,-0.2]],
                   [[0.6,0.2],[0.7,-0.3],[0.0,0.0]]]
chosen_mask     = [[1,1,1],[1,1,0]]      # 第二条末有效 token 在 idx1
rejected_mask   = [[1,1,0],[1,1,1]]

池化: hc=[chosen[0,2]=[0.5,0.2], chosen[1,1]=[0.9,0.4]]
      hr=[rejected[0,1]=[0.2,0.5], rejected[1,2]=[0.0,0.0]]
打分: r_c=[0.5+0.5*0.2+0.1=0.7, 0.9+0.2+0.1=1.2]
      r_r=[0.2+0.25+0.1=0.55, 0+0+0.1=0.1]
diff =[0.15, 1.1]
loss  = -mean(logsigmoid([0.15,1.1])) = 0.454146
```

## 约束

- CPU float32；每条序列至少有一个有效 token。
- "最后一个有效 token" 指下标最大的 mask$=1$ 位置（padding 通常在右侧，但不假设 mask 连续）。
- log-sigmoid 数值稳定；返回 0 维标量张量。
