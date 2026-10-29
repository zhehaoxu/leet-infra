# 详解：不可辨识的自由度必须被投影掉

## 出处

weight_norm 的反向是「重参数化求导」这一类问题的原型，也是 StyleGAN/Seq2Seq 时代留下的常备追问。同一个几何结构在今天的 LLM 工程里以另一种形式出现：**LoRA 的 \(B A\) 分解、谱归一化、以及任何「一个自由度被两个参数共享」的写法**，都要做同样的投影消歧。面试官用它来看你是否理解「梯度不唯一 ⟺ 参数化有冗余」。

## 推导：为什么 dV 必然与 v 正交

对单片 \(w=g\,v/\lVert v\rVert\)，最干净的做法是把 \(dv\) 分解成沿 \(\hat v\) 与正交两部分：沿 \(\hat v\) 的分量只改变 \(\lVert v\rVert\)，而它的作用与改变 \(g\) **完全等价**，因此在给定 \(dw\) 时不可分辨。令内积 \(\langle a,b\rangle=\sum_j a_jb_j\)，则
$$dg=\langle dW,\hat v\rangle,\qquad dV=\frac{g}{\lVert v\rVert}\big(dV_\perp\big)=\frac{g}{\lVert v\rVert}\big(dW-\hat v\,dg\big)$$
于是 \(\langle dV,v\rangle=\frac{g}{\lVert v\rVert}(\langle dW,v\rangle-\lVert v\rVert dg)=0\) 恒成立——这条正交性是判题里最锋利的自检。

## 与「直接对 W 求导」的区别

如果 autograd 沿着 `W = g * V / V.norm()` 这条计算图走，得到的 \(dV\) 会自动包含沿 \(v\) 的分量吗？不会：图求导给出的正是上式，因为 \(\partial\lVert v\rVert/\partial v=\hat v\) 那一项恰好把共线部分抵消。这就是为什么本题可以直接拿 autograd 对拍，也是「投影掉」这一说法的来历。

## 核心与易错点

- **折叠轴**：卷积权重 `(out, in, kh, kw)` 的 \(\lVert\cdot\rVert\) 要把除第 0 轴以外的所有轴折进去，`dim=` 写错会得到逐元素的错误归一。
- **零向量切片**：\(\lVert v\rVert=0\) 时方向无定义，PyTorch 的实现会加 `eps` 保护；本题按要求直接抛 ValueError。
- **\(dg\) 用 \(\hat v\) 而不是 \(v\)**：漏除 \(\lVert v\rVert\) 会让 \(dg\) 放大 \(\lVert v\rVert\) 倍。
- **dtype 分离**：`dg` 跟 `g`、`dV` 跟 `V`，两者可能不同（混合精度里常见）。

## 与邻近题的区别

`ag-layernorm-backward` 也含「减掉一个投影」（\(\hat x\) 方向），但那是归一化约束；本题的投影来自参数冗余，即使没有归一化轴也存在。`ag-conv2d-backward` 求的是裸权重的 \(dw\)，不涉及重参数化。

## 对拍口径与复杂度

oracle 用 `W = g.unsqueeze(1) * V / V.norm(dim=..., keepdim=True)` 建图后取 `[dg, dV]`，float64 atol=1e-9；另用「缩放某片 \(V\) 结果不变」与「\(dV\perp v\)」两条性质用例交叉验证。示例两片 \(\lVert v\rVert=5\)，四个数都能口算。复杂度 \(O(\text{numel})\)，每片两次归约。
