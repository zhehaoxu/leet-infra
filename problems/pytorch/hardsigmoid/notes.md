# Hard-Sigmoid 详解

## 定义

$$ \operatorname{hs}(x)=\operatorname{clip}\!\Big(\frac{x+3}{6},\,0,\,1\Big)=\frac{\operatorname{clip}(x+3,\,0,\,6)}{6}\in[0,1] $$

用一条折线（在 $[-3,3]$ 上线性、两端饱和）近似 Sigmoid，免去指数运算。

## 设计动机

Sigmoid 的 $e^x$ 在嵌入式 / 定点推理上昂贵。Hard-Sigmoid 只需一次加、一次截断、一次除，对 INT8 量化与移动端友好；输出范围与 Sigmoid 同为 $[0,1]$，可直接替换作门或激活。它与 ReLU6 本质同源：

$$ \operatorname{HS}(x)=\frac{\operatorname{ReLU6}(x+3)}{6},\qquad \operatorname{HardSwish}(x)=x\cdot\operatorname{HS}(x) $$

## 实现与数值要点

- 参考解 `torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0`，先平移再截断最后除。
- 常数 $3$ 与 $6$ 是**标准口径**（`nn.Hardtanh(-3,3)` 归一化到 $[0,1]$ 等价）；别用 $(x/2+0.5)$ 那种 MobileNet 论文里另一种 scale（截断点 $\pm2.5$），两者在 $|x|<3$ 段斜率不同，会对不齐。
- 折线在 $x=\pm3$ 处不可导，工程次梯度按饱和段 0 处理，判题不涉及梯度。

## 关联

HardSwish = $x\cdot$HardSigmoid；ReLU6 是其上界截断近亲；Sigmoid 是被近似的原函数。
