# 详解：softmax 反向：给定输出与上游梯度求 logits 梯度

## 出处

PyTorch 的 `aten::_softmax_backward_data` / `_log_softmax_backward_data`（`SoftMax.cu`、`SoftMax.cpp`）就是这个公式的两条支路；Goodfellow《Deep Learning》6.2.2 给出 softmax+交叉熵的合并推导，FlashAttention 论文（Dao 2022, 2205.14135）反向的第 3 步把它写成「用 O 与 dO 免存 P」的形式。面试里「手写 softmax 反向」是高频白板题。

## 从雅可比到一次点积

前向 \(y_i = e^{x_i}/\sum_k e^{x_k}\)，对 \(x_j\) 求导：

- \(i=j\)：\(\partial y_i/\partial x_i = y_i(1-y_i)\)
- \(i\ne j\)：\(\partial y_i/\partial x_j = -y_i y_j\)

合起来 \(\partial y_i/\partial x_j = y_i(\delta_{ij}-y_j)\)。链式法则：

$$dx_j=\sum_i grad_i\,y_i(\delta_{ij}-y_j)=y_j\Big(grad_j-\sum_i y_i\,grad_i\Big)$$

括号里的 \(\sum_i y_i grad_i\) 正是「grad 与 y 沿归一化轴的点积」，一个标量（每行一个），所以整条反向只是一次 `sum` 加一次逐元素乘，复杂度从 \(O(n^2)\) 降到 \(O(n)\)。

log_softmax 支路不同：前向 \(y_i=x_i-\log\sum_k e^{x_k}\)，直接对 \(x_j\) 求导得 \(\partial y_i/\partial x_j=\delta_{ij}-p_j\)（\(p=\exp y\)），于是 \(dx_j = grad_j - p_j\sum_i grad_i\)——减的是 **grad 的总和**乘以概率，而不是 y 与 grad 的点积。

## 核心与易错点

- 归约轴必须与 `dim` 一致，并用 `keepdim=True`（或 reshape）避免广播错轴；中间轴（如 `dim=1` 的 (B,C,N)）最容易退化成按末轴归约。
- 写成 `y*(grad - (grad*y).sum(dim))` 与 `y*grad - y*(y*grad).sum(dim)` 等价；漏掉外层 \(y_j\) 是常见手误，结果会整体少乘一次概率。
- 平移不变性是可用来自检的强性质：grad 沿 dim 为常数 ⇒ 结果恒为 0，因为 softmax 对 logits 加同一常数不变。
- log_softmax 支路把 \(e^{y_i}\) 当成 \(y_i\) 用，是本题最「像对」的错误：数值上只差一个指数，对拍时全错。
- 与交叉熵合写时 \(\frac{1}{C}\) 与「有效行数」的分母容易漏乘，但那属于 ag-cross-entropy-backward 的口径。

## 与邻近题的区别

- 与 ag-cross-entropy-backward：那题是「softmax + NLL」的合并结果 \((p-\text{onehot})/n\)，不需要上游 grad；本题保留任意 grad。
- 与 ag-attention-qkv-backward / ag-flash-attn-backward：那两题把本公式嵌进 \(QK^\top\) 的矩阵反向里（前者存 P，后者用 LSE 与 D=rowsum(dO∘O) 重建 P）。
- 与 operators 合集的 softmax / log-softmax：那是前向与数值稳定性（减最大值）；本题只有反向。

## 对拍口径与复杂度

判题在测试文件内独立重写前向（`torch.softmax` / `torch.log_softmax`）并用 `torch.autograd.grad` 取梯度作为 oracle，与参考解结构不同但数值一致；float64 用例 atol=1e-9。另有「grad 全 1 ⇒ 输出全 0」的平移不变性专项断言与 `log_input` 双支路交叉验证。时间 \(O(N)\)（一次归约 + 一次逐元素），额外空间一个 keepdim 标量。
