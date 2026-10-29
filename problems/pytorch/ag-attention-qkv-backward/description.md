# 注意力对 Q/K/V 的反向：掩码位零梯度与 GQA 折头求和

## 题目描述

注意力反向是 Transformer 训练里最贵的一段，也是面试里区分「写过 fused kernel」和「只调过 API」的分水岭。它有三个耦合的梯度：\(dV\) 只需一次带 \(P^T\) 的乘法，\(dQ/dK\) 却要先穿过 softmax——而穿过 softmax 的那个 \(\sum_k P_k\,dP_k\) 常被误写成 \(\sum_k dP_k\)。此外 GQA 让多个 Q 头共享一份 K/V，\(dK,dV\) 必须沿组做**求和**并返回 \(H_{kv}\) 而非 \(H_q\) 个头。本题给你 `Q, K, V, P, dO`，返回 `(dQ, dK, dV)`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- \(dP=dO\,V_{head}^T\)；沿 \(P\) 的末轴做 softmax 反向 \(dS=P\circ(dP-\text{rowsum}(P\circ dP))\)，最后再乘 `scale`。
- \(dQ=dS\,K_{head}\) 形状同 `Q`；\(dK=dS^T Q\)、\(dV=P^T dO\) 都要把同组 `groups` 个 Q 头**累加**，返回形状的头数是 `Hkv`。
- 掩码位上 \(P=0\) ⇒ 该处梯度必为 0，不得再额外施加 mask 或补常数。
- `Hq % Hkv != 0`、`K`/`V` 头数不等、任意形状不符一律抛 ValueError；返回 dtype 与 `Q` 一致（内部 float64 累计）。

## 示例

1 头、\(S_q=1\)、\(S_k=2\)、\(D=1\)、`scale=0.5`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
Q = [[[4.0]]]   K = [[[1.0], [2.0]]]   V = [[[1.0], [3.0]]]
P = [[[0.5, 0.5]]]   dO = [[[2.0]]]
dP = [2.0, 6.0],  rowsum(P*dP) = 4.0,  dS = P*(dP-4)*scale = [-0.5, 0.5]
dQ = [[[0.5]]]   dK = [[[-2.0], [2.0]]]   dV = [[[1.0], [1.0]]]
```

\(dQ=-0.5\cdot1+0.5\cdot2=0.5\)；\(dK=dS^T Q=[-0.5\cdot4,\,0.5\cdot4]\)。

## 约束

- `P` 是前向已算好的概率（每行和为 1），不接受 logits；内部 float64 累计后转回 `Q.dtype`
- 与「用 autograd 搭 `softmax(QK^T*scale) @ V`」的 oracle 对拍，float64 atol=1e-9；GQA 用例 `Hq = 2·Hkv`
