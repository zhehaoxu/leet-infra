# 详解：缩放放在前向还是后向，决定反向长什么样

## 出处

dropout 反向本身简单，但它是「原样 dropout vs inverted dropout」这道概念题的最佳载体，也是训练/推理不对称（`model.train()` 与 `model.eval()`）最常被抓出来的地方：推理框架如果把训练态的 \(1/(1-p)\) 带进 eval 图，输出会整体偏大。做过 ONNX 导出或推理对齐的人对此有肌肉记忆。

## 两种约定的等价性

旧式（vanilla）dropout 前向 \(y=x\cdot mask\)、反向 \(dx=dY\cdot mask\)，把 \(1-p\) 的补偿推迟到推理时缩权重；inverted dropout 前向 \(y=x\cdot mask/(1-p)\)，对 \(x\) 求导得到 \(\partial y/\partial x=mask/(1-p)\)，因此**反向也带一个 \(1/(1-p)\)**。两者期望一致，但缩放搬运的位置不同。PyTorch 用后者，实测 `F.dropout(x, 0.5)` 的反向值域是 \(\{0,2\}\) 而不是 \(\{0,1\}\)；把两种约定混用就会多除或少除一次，判题按 `F.dropout` 对拍。

## train=False 的完全恒等

推理态 dropout 是恒等映射，因此反向也是恒等：`mask` 可以是任意形状（甚至全 0）、`p` 可以是 0，结果都必须等于 `dY`。这是测试里最容易漏的一条——很多实现把「除 \(1-p\)」写在分支外面，`train=False` 时也放大，输出直接错 \(1/(1-p)\) 倍。

## 核心与易错点

- **广播掩码**：`dropout2d` 的 mask 形状是 `(1,C,1,1)`，要 `expand` 到 `dY` 形状而不是要求同形；不可广播时必须抛 ValueError 而不是静默按元素错配。
- **`p=0` 是合法的**：此时缩放是 1，等价于恒等，但仍要走训练态分支（用 mask）。
- **`mask` dtype**：bool 张量不能直接与 float 相乘（PyTorch 会隐式转，但要显式 `.to(float64)` 才保证与 autograd 逐位一致）。
- **不要重复缩放**：如果前向由调用方完成，反向只能做一次 \(1/(1-p)\)。

## 与邻近题的区别

`ag-activation-backward` 是纯逐元素解析导数、与随机性无关；`ag-embedding-scatter-backward` 也有掩码，但那是语义常量（padding）而不是采样结果。本题的独有考点是「模式开关决定公式形状」。

## 对拍口径与复杂度

oracle 用 `F.dropout(x, p, training=True)` 前向 + `torch.autograd.grad`，并把前向同一次采样的 mask 传进来比对，float64 atol=1e-9。示例 \(p=0.5\) 让缩放是整数 2，四格数值可一眼核对。复杂度 \(O(\text{numel})\)，仅一次逐元素乘除。
