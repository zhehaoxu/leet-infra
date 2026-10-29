# 详解：为什么反向只需要 O 和 LSE

## 出处

FlashAttention（Dao et al., 2022）的反向是推理/训练框架岗的高频追问：「你说不出省了哪块内存，就等于没读懂这篇论文」。面试的标准问法是：前向不存 \(P\)，反向怎么把 \(dP\) 穿过 softmax？答出 \(D=\text{rowsum}(dO\circ O)\) 这一行就算过关。

## 把 softmax 反向改写成只含 O、dO 的形式

朴素反向要算 \(dS=P\circ(dP-\sum_k P_k dP_k)\)，其中 \(\sum_kP_kdP_k=\sum_k P_{ij}(dO_i\cdot V_j)=dO_i\cdot\sum_jP_{ij}V_j=dO_i\cdot O_i\)。也就是说那一行归约**恰好等于前向已经算好的 \(O\)** 与 \(dO\) 的点积：

$$D_i=\sum_d dO_{id}O_{id},\qquad dS_{ij}=P_{ij}(dP_{ij}-D_i)\cdot scale$$

于是反向只需重算 \(P\)（用 \(S+mask-LSE\)，逐块 tile 内即可，不必存整个 \(P\)），配合 \(O,dO\) 就能给出全部三段梯度。\(LSE\) 的角色是把 \(\exp\) 直接归一化：若只减行最大值，还得再除一次 \(\sum\exp\)，而那份和已经被 \(LSE\) 吸收了。

## 核心与易错点

- **`mask` 必须同时进 \(S\) 与 \(LSE\)**：前向的 \(LSE\) 是含掩码的 logsumexp，重算时也要 \(S+mask-LSE\)；只在一处加掩码会让整行概率不再归一，`causal` 用例立刻错。
- **`D` 用 `dO·O` 而不是 `dO·(某个常数)`**：测试用「\(dO\) 行和为 0 时平移 \(O\) 结果不变、\(dO\) 全 1 时结果必须变」这对双向断言把它钉死。
- **`LSE` 单独加常数 \(c\)** ⇒ \(P\) 缩放 \(e^{-c}\) ⇒ 三个梯度同比例缩放；这是检查归一化是否走对路径的最便宜手段。
- **数值范围**：\(S+mask-LSE\) 理论上 \(\le 0\)，`-inf` 掩码位给 `exp(-inf)=0`；先减 LSE 再 exp 保证不上溢。

## 与邻近题的区别

`ag-attention-qkv-backward` 直接给你 \(P\)，不需要重算、也不需要 \(D\)（归一化藏在 \(P\) 里）；本题把 \(P\) 藏起来，只给 \(O,LSE\)，考察的正是那句等价改写。因果性、分块的内存调度不在本题范围内。

## 对拍口径与复杂度

oracle 用测试内独立前向 `softmax((Q@Kᵀ)*scale+mask) @ V` 配 `torch.autograd.grad`，float64 atol=1e-9；示例 \(S=[0,0]\)、\(LSE=\log 2\) 使所有量为 \(\pm0.25/0.5\) 的精确十进制。复杂度仍是 \(O(n^2d)\)（本实现不做真分块），但**峰值额外内存**从「存 \(B H n^2\) 的 \(P\)」降到「存 \(B H n\) 的 \(LSE,D\)」。
