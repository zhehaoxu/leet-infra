# FlashAttention 反向：用 O、LSE、dO 逐块重算梯度

## 题目描述

FlashAttention 的核心承诺是「不落地 \(n\times n\) 的概率矩阵」：前向只保留输出 \(O\) 与每行的 log-sum-exp \(LSE\)。反向就必须**重算** \(P=e^{S+mask-LSE}\)，并且用一个每行标量 \(D=\text{rowsum}(dO\circ O)\) 顶替原本藏在 \(P\) 里的归一化信息。本题给你 `Q, K, V, O, LSE, dO`（可选 `mask`），要求返回 `(dQ, dK, dV)`，且任何时刻都不构造完整概率矩阵以外的显式 \(n\times n\) 存储依赖。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 重算用 \(P=\exp(S+mask-LSE)\)，**减的是 LSE 而不是行最大值**；掩码位因此自动得 0。
- \(dV=P^T dO\)；\(dP=dO\,V^T\)；\(D=\text{rowsum}(dO\circ O)\)；\(dS=P\circ(dP-D)\cdot scale\)；\(dQ=dS\,K\)，\(dK=dS^T Q\)。
- 漏掉 \(D\) 等价于忘了 softmax 反向的减均值项：\(dQ,dK\) 会整体错掉，而 \(dV\) 仍然正确。
- `LSE` 形状必须是 `(B, H, Sq)`，`mask` 必须可广播到 `(B, H, Sq, Sk)`；任何形状不符抛 ValueError。内部 float64，返回 dtype 随 Q/K/V。

## 示例

1 头、\(S_q=1\)、\(S_k=2\)、\(D=2\)、`scale=1.0`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
Q = [[[0, 1]]]  K = [[[1, 0], [0, 0]]]  V = [[[0, 0], [1, 1]]]
S = [0, 0] ⇒ P = [0.5, 0.5], LSE = log 2, O = P@V = [0.5, 0.5], dO = [1, 0]
dP = dO@Vᵀ = [0, 1],  D = dO·O = 0.5,  dS = P∘(dP−D) = [−0.25, 0.25]
dQ = [[[−0.25, 0]]]   dK = [[[0, −0.25], [0, 0.25]]]   dV = [[[0.5, 0], [0.5, 0]]]
```

## 约束

- 与「朴素 `softmax(QKᵀ·scale+mask) @ V` + autograd」的 oracle 逐位对拍，float64 atol=1e-9
- 性质用例：单独把 `LSE` 加常数 \(c\) ⇒ 三个梯度整体缩放 \(e^{-c}\)；`mask` 与 `LSE` 同加 \(c\) ⇒ 结果不变；\(dO\) 行和为 0 时平移 \(O\) 不改变结果
