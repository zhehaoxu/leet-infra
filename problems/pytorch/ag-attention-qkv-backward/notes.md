# 详解：注意力反向的 softmax 折算与 GQA 归并

## 出处

「写出 scaled-dot-product attention 的反向」是大模型框架岗（训练 kernel、推理引擎、算子融合）出现频率最高的手推题之一，FlashAttention 论文的原稿附录即为此式；GQA 折头求和则是 LLaMA-2/3、Qwen 一系实现里必须处理的一条分支。

## dS = P∘(dP − Σ P∘dP) 从哪来

前向 \(O_i=\sum_j P_{ij}V_j\)，\(P_{ij}\) 是第 \(i\) 行 logits 的 softmax。由 softmax 反向（见 `ag-softmax-backward`）：

$$\frac{\partial L}{\partial s_{ij}}=P_{ij}\Big(\frac{\partial L}{\partial P_{ij}}-\sum_k P_{ik}\frac{\partial L}{\partial P_{ik}}\Big),\qquad \frac{\partial L}{\partial P_{ij}}=dO_i\cdot V_j$$

而 \(\partial L/\partial s\) 还要乘前向的 `scale`（\(s_{ij}=\text{scale}\cdot q_i\cdot k_j+\text{mask}\)）才能变成对 \(QK^T\) 的梯度。把这两步混为一谈（漏 `scale`）时，\(dQ,dK\) 会整体差一个 \(1/\sqrt D\)，\(dV\) 却完全正确——这就是漏乘 `scale` 的定位方法。

## GQA 的折头

\(dK\)、\(dV\) 的头轴长度是 \(H_{kv}\)。对每个 kv 头 \(t\)，把属于它的 \(H_q/H_{kv}\) 个 q 头贡献相加：\(dK^{(t)}=\sum_{h:\,h//g=t} dS_h^TQ_h\)。实现上可 reshape 成 `(B,Hkv,g,Sk,D)` 再对 `g` 轴求和，或直接 `index_add`。**忘记折头**会得到 `(B,Hq,Sk,D)`，与 autograd 的形状都不一致；**折头时把 `groups` 与 `Hkv` 弄反**（reshape 成 `(B,g,Hkv,...)`）则形状对但数值错，是最阴的一种 bug。

## 核心与易错点

- **`rowsum(P∘dP)` 不能写成 `rowsum(dP)`**：掩码行（`P` 有零）时两者立刻分叉。
- **掩码位天然给 0**：\(P_{ij}=0\) 使 \(dS_{ij}=0\)，无需再乘 mask；如果额外乘了 mask 的补集，反而会把合法位置清零。
- **`dV` 不过 softmax**：\(dV=\sum_h P_h^T dO_h\)，与 `scale` 无关。
- **广播检查**：`K`/`V` 头数为 `Hkv` 时，与 `dO` 相乘前必须按 `h // groups` 取头，不能靠广播混过去。

## 与邻近题的区别

`ag-flash-attn-backward` 拿到的是 `O, LSE, dO` 而没有 `P`，必须先重算 \(P=e^{s-LSE}\)；本题直接给 `P`，考察点集中在 softmax 折算与折头。同集的 `additive-attention-mask` 只处理掩码构造、不涉及求导。

## 对拍口径与复杂度

oracle 用 autograd 搭 `P = softmax(Q @ K_head^T * scale)`、`O = P @ V_head`，以 `loss = (O*dO).sum()` 取三个梯度，float64 atol=1e-9。示例是 1 头 1 查询的标量情形，四个数值可完全手算。复杂度 \(O(BH_qS_qS_kD)\)（两次批量乘法 + 一次行归约）。
