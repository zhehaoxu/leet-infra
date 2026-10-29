# 详解：三种池化的路由与除数

## 出处

池化反向常见于视觉/多模态岗位的手写题，也常作为「你理解的 autograd 到底是什么」的引子：`max` 的梯度是**不可微点的次梯度**，`adaptive_avg` 则考验能否把 PyTorch 那套 `start_index = floor(i·H/OH)` 的窗口定义准确复现。做推理对齐的人必须会写，因为导出 ONNX 时这两条要手动展开。

## max：路由与并列约定

\(\partial y_{ij}/\partial x_{pq}=\mathbb 1[(p,q)=\arg\max_{\text{window}}x]\)，所以 \(dx\) 是「把 \(dY\) 按 argmax 散射回去」，实现上要么记录前向的索引（`return_indices=True`），要么在窗口内重新比较。并列时 PyTorch 的 CUDA/CPU 行为并不完全一致，本题固定为**行主序最小**（即 `argmax` 取第一个最大值），这样判题才是确定的。用 `scatter` 时要把索引展平成 `(B,C,OH,OW)` 的一维下标，`dim` 弄错就会把梯度撒到别的通道。

## avg 与 adaptive_avg 的除数差别

固定窗口 `avg` 的除数恒为 \(k_hk_w\)（`padding=0` 时），而 adaptive 的窗高 \(w_i=\lceil(i+1)H/OH\rceil-\lfloor iH/OH\rfloor\) 随 \(i\) 变化，且当 \(OH\nmid H\) 时窗口会**重叠**：同一个位置可能属于两个窗口，于是它拿到的是两份 \(dY/w_iw_j\) 之和。「一律除以 kernel 面积」在这种输入下必错，测试专门放了 \(H=5,OH=2\) 这类不整除的形状。

## 核心与易错点

- **`stride` 与 `kernel` 分离**：默认 `stride=kernel`，但显式传入更小的 stride 时窗口重叠，`max` 的 argmax 路由与 `avg` 的累加都要按 `stride` 走。
- **`padding` 只对 `max` 合法**：pad 区的虚拟位置没有对应的 \(x\) 元素，`avg` 的除数会因此变得含糊，本题直接抛 ValueError。
- **dx 形状恒等于 x**：不要返回去掉 padding 的尺寸。
- **dtype**：`x` 是整数也要返回浮点（与入参 dtype 一致，内部 float64 计算）。

## 与邻近题的区别

`ag-conv2d-backward` 同时求 \(dx\) 与 \(dw\) 且核可学；`adaptive-avgpool`/`adaptive-maxpool` 是本库里的**正向**题，只要求算输出与窗口边界。本题只处理反向路由。

## 对拍口径与复杂度

oracle 分别用 `F.max_pool2d`、`F.avg_pool2d`、`F.adaptive_avg_pool2d` 前向后 `torch.autograd.grad((y*dY).sum(), x)`，float64 atol=1e-9。示例用严格递增的 \(4\times4\)，四个 argmax 位置肉眼可定。复杂度 \(O(BCHW\cdot k_hk_w)\)（窗口枚举）或 \(O(BCHW)\)（索引散射）。
