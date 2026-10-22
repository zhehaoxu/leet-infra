# PReLU 详解

## 定义

$$ \text{prelu}(x)=\max(0,x)+a\cdot\min(0,x) $$

即 Leaky-ReLU，但负斜率 $a$ 是**可学习参数**。

## 为什么让斜率可学

- Leaky-ReLU 的负斜率是超参（如 0.01）；PReLU 把它变成每通道可训练，Deep PReLU Networks（He 等）在 ImageNet 上借此超过 ReLU。
- 初始化 $a=0.25$ 是论文推荐值；$a\to0$ 退化为 ReLU，$a\to1$ 退化为恒等，$a=0.01$ 即 Leaky-ReLU。

## 实现与广播要点

- $a$ 通常形状为 `(C,)`（每通道一个），需按通道广播：对 $(N,C,H,W)$ 要 reshape 成 $(1,C,1,1)$ 再乘负半部分。本关按给定 $a$ 的广播语义实现。
- 分解成 `torch.where(x>0, x, a*x)`，或 `relu(x)+a*relu(-x)`，两种等价；注意 $x=0$ 处输出恒 0。

## 梯度（理解用）

对输入：$x>0$ 为 1，否则 $a$。对参数 $a$：梯度是负半部分输入之和 $\sum_{x<0}x$——所以 $a$ 会往「让负值别太负」的方向更新。
