# im2col：把卷积变成矩阵乘

## 核心思想

卷积 = 在每个输出位置做一次「感受野 · 核」点积。im2col 把所有输出位置的感受野**展开成列向量**排成大矩阵，于是整层卷积退化为**一次 GEMM**：

$$ \underbrace{\text{cols}}_{(N,\;o\!h\cdot o\!w,\;c_{in}k_hk_w)}\times \underbrace{\text{rows}}_{(c_{in}k_hk_w,\;c_{out})}\;\Rightarrow\;(N,o\!h\cdot o\!w,c_{out})$$

这是 cuDNN/BLAS 加速卷积的原理：用内存换掉低效的四重循环，复用高度优化的矩阵乘。

## cols 的构造（本关关键）

- `x.unfold(2,kh,sh).unfold(3,kw,sw)` → $(N,C,nh,nw,kh,kw)$，最后两维是每个窗口的感受野。
- permute 成 $(N,nh,nw,C,kh,kw)$ 后 reshape $(N, oh\cdot ow, c_{in}k_hk_w)$：**行=输出位置，列=感受野展平**。列的顺序必须与权重展平 `weight.reshape(cout, cin*kh*kw)` 的维度顺序一致（C 外层、kh、kw 内层），否则乘错。

## 与权重的对齐

`rows = weight.reshape(cout, cin*kh*kw).T`，形状 $(c_{in}k_hk_w, c_{out})$。cols 的列维和 rows 的行维都是 $c_{in}k_hk_w$，同序点积才成立。

## 变体

- **im2row**：行列转置写法（感受野当行）。
- 3D 卷积对应 volume2col。
- 反面 **col2im**：反向传播把「各列的梯度」累加回输入，重叠位置要求和——这也是为何反向不能简单 reshape。

## 尺寸

先手动 0-pad（`F.pad`，注意末维优先），$oh=\lfloor(h-kh)/sh\rfloor+1$。im2col 只处理 $g=1,d=1$；分组卷积要分块做、膨胀卷积把核元素插零（见「膨胀 unfold」关）。
