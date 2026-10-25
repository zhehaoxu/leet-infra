# PEFT：Adapter 瓶颈前馈与残差

## 题目描述

**Adapter**（Houlsby et al., arXiv:1902.00751）在冻结的 Transformer 层之间插入一个小小的「瓶颈-扩回」前馈模块：先把 \(d\) 维降采样到 \(r\)（\(r\ll d\)）、过非线性、再升回 \(d\)，最后**加回原输入**（残差）。微调时只训这些旁路小模块，主干全部冻结。

单条 Adapter 的前向：

\[y = W_\text{up}\,\sigma\!\bigl(W_\text{down}\,x\bigr) + x\]

\(\sigma\) 取 ReLU。降维幅度 \(d/r\) 就是「参数efficiency」的来源，残差保证插入瞬间近似恒等。

## 规则

给定输入 `x` 形状 \((\dots, d)\)、降维 `W_down` 形状 \((r, d)\)、升维 `W_up` 形状 \((d, r)\)：

1. 下投影 \(h = x\,W_\text{down}^\top\)（\((\dots, r)\)）。
2. 非线性 \(a = \operatorname{ReLU}(h)\)。
3. 上投影 \(u = a\,W_\text{up}^\top\)（\((\dots, d)\)）。
4. 残差 \(y = u + x\)。

返回 `y`（形状 \((\dots, d)\)）。全程 float32，禁止调用现成 PEFT 库；为判题确定性**不加** Dropout。

## 示例

`x`（2×3）\(=\begin{bmatrix}0.61&1.62&-0.80\\ 0.52&-0.21&-0.07\end{bmatrix}\)、`W_down`（2×3）\(=\begin{bmatrix}0.4&-0.2&0.1\\ 0.0&0.3&-0.5\end{bmatrix}\)、`W_up`（3×2）\(=\begin{bmatrix}0.2&0.5\\ -0.3&0.1\\ 0.4&-0.2\end{bmatrix}\)（\(d=3,r=2\)）：

```
y = [[ 1.053,  1.709, -0.977],
     [ 0.569, -0.283,  0.027]]
```

## 约束

- CPU float32；`x` 支持 2D 与 3D（最后维为 \(d\)）。
- 残差要求 \(W_\text{up}\) 输出维与输入维同为 \(d\)，故 `W_up` 形状 \((d,r)\)。
- `W_up=0` 时上投影恒为 0、\(y=x\)（纯残差旁路）。
