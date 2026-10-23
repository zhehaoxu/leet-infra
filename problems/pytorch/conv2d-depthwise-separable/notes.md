# 深度可分离卷积详解

## 为什么拆开更省

标准 $3\times3$ 卷积计算量 $\propto C_{in}\cdot C_{out}\cdot k^2$。拆成两步：

- **depthwise**：$C_{in}$ 个通道各用一个 $k\times k$ 核 $\Rightarrow C_{in}\cdot k^2$；
- **pointwise**：$1\times1$ 稠密混合 $\Rightarrow C_{in}\cdot C_{out}$。

合计 $C_{in}(k^2 + C_{out})$，比稠密卷积少约 $k^2$ 倍（$k=3$ 时省约 8–9 倍参数与 FLOPs）。
MobileNet 的「宽度乘子」正是在这两步上再按比例砍通道。

## 关键点：depthwise 不跨通道、pointwise 不跨空间

- depthwise 权重形状 $(C,1,k,k)$，等价 `groups=C` 的卷积；**通道之间完全隔离**，
  只用 `pw` 里某一行的 $1\times1$ 权重就能验证（扰动其它通道不应改变输出）。
- pointwise 权重 $(C_{out},C,1,1)$，纯通道线性组合、$3\times3$ 空间关系保持不变。
- 二者顺序不可颠倒；缺了 pointwise 就无法改变通道数、通道间永不交互。

## 保尺寸与奇数核

题面用 `padding=kh//2`、`stride=1` 让 depthwise 输出与输入同尺寸（`same` 语义）。
因此约定 **kh 为奇数**——偶数核时 `kh//2` 无法让四边对称，尺寸会偏。

## 延伸

深度可分离思想也进入 Transformer：相对位置注意力里把「按位置卷积」拆成逐头 depthwise +
$1\times1$； separable conv 也是很多实时/移动端 backbone（MobileNetV1→V3、Xception、EfficientNet）
的基本单元。
