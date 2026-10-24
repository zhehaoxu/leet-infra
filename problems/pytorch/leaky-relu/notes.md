# Leaky ReLU 详解

## 定义

$$ \operatorname{leaky}(x)=\begin{cases} x, & x>0\\[3pt] a\,x, & x\le 0\end{cases}\qquad(a=\text{negative\_slope},\ \text{默认 }0.01) $$

给 ReLU 的负半轴加一个很小斜率，折线在原点连续。

## 设计动机

ReLU 负区梯度恒 0 → 死神经元。Leaky-ReLU 让负区仍保留一点梯度流（斜率 $a$），神经元一旦误入负区仍有机会「复活」。$a$ 通常取 $0.01$；也有 $1/3$、$0.1$ 等经验值。

## 实现与数值要点

- 参考解 `torch.where(x >= 0, x, negative_slope * x)`，向量化。
- **口径**：分界用 $x\ge0$（含原点走正值支，即原点输出 0）；这与 `torch.maximum` 的 ReLU 在原点结果一致，判题无歧义。
- 等价写法：`x * where(x>0, 1, a)`，或 `F.leaky_relu(x, negative_slope)`。
- $a$ 是显式入参，务必用它而不是硬编码 0.01，否则默认测试外的自定义斜率会错。

## 家族定位

- PReLU：把 $a$ 变成**可学习**参数（甚至逐通道不同）。
- ELU：负区改指数饱和到 $-a$，比线性更平滑、均值更接近 0。
- RReLU：训练时随机采样 $a$，作正则。

## 关联

题中 negative_slope 为标量参数；SELU 是 ELU 的自归一化定缩放版，别混淆二者斜率常量。
