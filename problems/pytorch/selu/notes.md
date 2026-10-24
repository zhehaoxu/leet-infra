# SELU 自归一化激活详解

## 曲线：带缩放的 ELU

SELU 就是给 ELU 乘一个 $\lambda$：正半轴斜率变成 $\lambda\approx1.05$，负半轴保留饱和的
$-\lambda\alpha$ 下界（$\approx-1.758$）。它在 $x=0$ 处连续（两侧都趋于 0）、可导，比 ReLU
多了负区的平滑梯度，比 ELU 多了整体缩放。

$$ \operatorname{SELU}(x)=\lambda\begin{cases}x,&x>0\\ \alpha(e^x-1),&x\le0\end{cases} $$

## 「自归一化」从哪里来

Klambauer et al. 2017 证明：若输入近似高斯、权重用 **SELU 专用初始化**（$\text{lecun_normal}$，
方差 $1/n$ 的正态），则这组 $(\lambda,\alpha)$ 恰是「不动点」——每层输出仍均值 0、方差 1。
于是深层网络无需 BN/LN 也能稳住激活分布，梯度不易爆炸/消失。

## 使用的前提与坑

自归一化是**一整套方案**，光换激活不够：
- 权重必须 `nn.init.lecun_normal_`（不是默认 kaiming/xavier）；
- 输入需先标准化到 mean0/var1；
- **不能随意 dropout**（会破坏不动点），要用 **AlphaDropout**（保持均值方差不变的专用变体）。
任一条件被破坏，自归一化优势就打折——这也是 SELU 没能像 ReLU 那样通用的原因。

## 与其它题呼应

- 与 `elu` 只差一个 $\lambda$，可对比着看缩放如何改变分布不动点。
- 与归一化类算子（layernorm/batchnorm）互为替代思路：一个靠外部 norm 层，一个把 norm
  内化进激活 + 初始化。
