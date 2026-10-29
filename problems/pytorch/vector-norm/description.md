# 向量范数 vector_norm（ord 全族与多轴归约）

## 题目描述

`torch.linalg.vector_norm` 看着简单，`ord` 一栏却埋着整片口径差异：默认 2-范数、`0` 是数非零元素、`±inf` 取绝对值的极值、`1` 是绝对值和而 `-1` 是调和式 $(\sum|x_i|)^{-1}$ 的倒数（只要段里有 0 就整体为 0），有限 $p$ 还要防上溢下溢。梯度裁剪、注意力权重归一、KV 重要性打分都直接踩在这套定义上。

$$\|x\|_p = \Big(\sum_i |x_i|^{\,p}\Big)^{1/p}, \qquad \|x\|_\infty = \max_i |x_i|, \qquad \|x\|_0 = \#\{i : x_i \ne 0\}$$

对有限 $p$，判题要求用**先提出最大绝对值**的稳定式

$$\|x\|_p = m\Big(\sum_i (|x_i|/m)^p\Big)^{1/p}, \qquad m = \max_i |x_i|$$

否则 $x\sim 10^3$ 且 $p=8$ 时 $\sum|x|^p$ 直接上溢成 inf。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `dim=None` 表示把整个张量展平后归约，返回 **0 维**张量；`dim` 可以是 int 或两根以上的 int 元组，负数按 `x.dim()` 折算。
- `ord` 接受 `None`（等价 2）、int、float、`"inf"` / `"-inf"` 字符串；`"fro"`、`"nuc"` 是矩阵专属，对向量无意义，须抛 ValueError。
- `dim=None` 且 `keepdim=True` 时，本题口径是**所有轴都保留为长度 1**（torch 在此情形返回 0 维，本题作扩展口径）。
- 全零向量在任何 `ord` 下都是 0；`ord=-1` 遇到 0 元素时结果是 0 而不是 inf。

## 示例

$X$ 为 $2\times 3$，沿 `dim=1` 各 `ord` 的结果（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
X = [[ 3, -4,  0],
     [ 0,  0, -1]]

ord=2    -> [5.0, 1.0]
ord=0    -> [2.0, 1.0]      # 非零元素个数
ord="inf"-> [4.0, 1.0]      # max|x|
dim=None -> 5.099019527435303   # sqrt(26)
```

第一行 $\|\cdot\|_2=\sqrt{3^2+4^2+0^2}=5$；`ord=0` 只数非零，$3$ 与 $-4$ 两个；`ord="inf"` 取 $\max(|3|,|4|,|0|)=4$。

## 约束

- 输入 float32，内部 float64 累计，返回 float32；与 `torch.linalg.vector_norm` 对拍 atol=1e-5、rtol=1e-5
- 形状与 dtype 一并断言；`ord`/`dim` 非法取值（越界轴、重复轴、空元组、矩阵记号）逐个断言 ValueError
