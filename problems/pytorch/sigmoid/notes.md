# Sigmoid 详解

## 定义

$$ \sigma(x)=\frac{1}{1+e^{-x}}\in(0,1),\qquad \sigma'(x)=\sigma(x)\,(1-\sigma(x)) $$

把任意实数压到 $(0,1)$，形状是关于原点对称的 S 曲线；$\sigma(-x)=1-\sigma(x)$。历史上用作二分类输出层与「门」的开关。

## 实现与数值要点

朴素 `1/(1+torch.exp(-x))` 在 $x$ 很负时 $e^{-x}$ **上溢成 inf**，结果虽仍趋 0 但伴随 overflow 警告/NaN 风险。参考解按符号分支保证指数参数始终非正：

- $x\ge 0$：$\dfrac{1}{1+e^{-x}}$（指数里是 $-x\le0$）。
- $x<0$：$\dfrac{e^{x}}{1+e^{x}}$（指数里是 $x<0$）。

两式代数相同，只是把会溢出的那一支改写。`torch.sigmoid` 内部即这样稳定。

## 梯度消失

$|x|$ 稍大即饱和：$\sigma\to0/1$ 时导数 $\to0$（峰值仅 0.25 在 $x=0$）。深层网络堆叠 sigmoid 会把梯度压扁——这正是 ReLU 取代隐藏层 sigmoid 的主因。今日 sigmoid 主要活在**门控**（LSTM 的 $i,f,o$、GRU、注意力 soft-cap）与概率输出。

## 关联

tanh $=2\sigma(2x)-1$；Hard-Sigmoid 是其分段线性廉价近似；Gumbel-Softmax、LoRA 缩放等都以 sigmoid 为构件。
