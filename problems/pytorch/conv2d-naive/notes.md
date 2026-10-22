# Conv2d（朴素实现）详解

## 定义：其实是「互相关」

深度学习里的「卷积」不做核翻转，是**互相关**：输出位是感受野与核的逐元素乘加：

$$ \text{out}(n,o,i,j)=\sum_{c}\sum_{p,q} x(n,c,\,i\cdot s+p,\,j\cdot s+q)\cdot w(o,c,p,q)+b(o) $$

torch 的 `F.conv2d`、cuDNN 也都是互相关，所以朴素实现对齐 `F.conv2d` 即可，**别去翻转核**。

## 形状约定（最容易搞反）

- 输入 `x`: $(N, C_{in}, H, W)$
- 权重 `weight`: $(C_{out}, C_{in}, k_h, k_w)$ —— 注意是 Cout 在前，与输入通道维度不同，别写反。
- 输出: $(N, C_{out}, H_{out}, W_{out})$，$H_{out}=\lfloor (H+2p-k_h)/s\rfloor+1$。

## padding 顺序坑

`F.pad` 的 pad 元组是**从末维往前、每维 (左,右)**：`(pw,pw,ph,ph)` 对应 W 左右、H 上下。写错顺序会把行列 padding 弄反。卷积里等价可用 `F.conv2d(..., padding=ph)`。

## 分组 / 膨胀（扩展视野）

- `groups=g`：$C_{in}$ 与 $C_{out}$ 各分 g 组，组内相连（Depthwise 即 $g=C_{in}$、$C_{in\_per\_group}=1$）。
- `dilation=d`：核元素按间隙采样，感受野变大而参数不变，$H_{out}=\lfloor (H+2p-d(k-1)-1)/s\rfloor+1$。
本关是 $g=1,d=1$ 的最简形；理解后再看「im2col」「膨胀 unfold」两关。

## 朴素 vs 高效

四重循环是教学版；工业实现走 im2col+GEMM 或直接 GEMM/Winograd，把卷积转成矩阵乘复用 BLAS 吞吐（见「im2col」「批量矩阵乘」）。
