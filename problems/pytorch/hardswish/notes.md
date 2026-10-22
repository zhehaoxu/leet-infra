# HardSwish 详解

## 定义

$$ \text{hardswish}(x)=x\cdot\frac{\text{ReLU6}(x+3)}{6} $$

其中 $\text{ReLU6}(t)=\min(\max(t,0),6)$。

## 为什么有它

Swish/SiLU 要算 sigmoid（含指数），在移动端/整型量化推理里不友好。HardSwish 用**分段线性**逼近 Swish，去掉指数，硬件更省，是 MobileNetV3 的激活。

$$ \text{hardswish}(x)\approx \text{swish}(x), $$

但只在 $[-3,3]$ 之间是斜率渐变的折线，两端：$x\le-3\to0$，$x\ge3\to x$。

## 实现要点（边界最容易错）

- clip 的是 $x+3$ 到 $[0,6]$ 再除 6，等价于门 $g(x)$：
  - $x<-3$：$g=0$，输出 0
  - $-3\le x\le 3$：$g=(x+3)/6$，输出 $x(x+3)/6$
  - $x>3$：$g=1$，输出 $x$
- 用 `x * torch.clamp(x+3, 0, 6) / 6`。注意分母是 6，别写成 `F.hardswish` 里另一个常数版本（PyTorch 官方 `F.hardswish` 用的门是 $\text{ReLU6}(x+3)/6$，与本关一致）。

## 与 HardSigmoid

HardSigmoid $=\text{ReLU6}(x+3)/6$ 就是这里的门；HardSwish $=x\cdot$HardSigmoid。两者差一个 $x$ 因子。
