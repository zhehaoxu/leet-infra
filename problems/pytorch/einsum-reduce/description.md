# einsum 表达式求值器（标注串 → 乘加归约）

## 题目描述

`torch.einsum` 用一个「下标串」把 matmul、batch matmul、转置、外积、按轴求和全部统一成同一种写法：字母给每个轴起名，出现在多个操作数里的字母是 batch/共享轴，出现在两个操作数里但被「消掉」的字母是收缩轴，输出里不写的字母就是被求和归约掉的轴。注意力代码里 `torch.einsum("bhqd,bhkd->bhqk", q, k)` 就是它最常见的使用场景。

$$\text{out}[\mathcal{O}] = \sum_{c \in \mathcal{C}} \prod_{k} T_k[\text{按标注对齐的切片}]$$

其中 $\mathcal{C}$ 是「出现在输入但未出现在输出」的字母集合（收缩轴），$\mathcal{O}$ 是输出标注串。本题实现该语法的受限子集：显式 `->`、1–3 个操作数、字母不重复（不支持 trace/diagonal）、不支持省略号。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 语义必须与 `torch.einsum` 在支持的子集内逐点一致：`"ij,jk->ik"` 等于矩阵乘、`"bij,bjk->bik"` 等于 batch 矩阵乘、`"bhqd,bhkd->bhqk"` 等于注意力打分、`"ij->ji"` 等于转置、`"ij->i"` 等于按行求和、`"i,i->"` 等于点积。
- 输出字母若全部被收缩（如 `"i,i->"`）返回 **0 维**张量；`"ij->ji"` 这类纯置换没有收缩轴。
- 一个字母若在某个操作数里出现两次（`"ii->i"`），那是对角/迹操作，本题**不支持**，须抛 ValueError；同理 `"..."`、缺 `->`、输出字母在输入中不存在、标注字母数与张量维数不符、操作数个数超过 3 都抛 ValueError。
- 内部可用 float64 累计，返回值 dtype 与第一个操作数相同（float32）。

## 示例

$X$ 为 $3\times 2$、$Y$ 为 $2\times 3$，spec 取 `"ij,jk->ik"`（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
X = [[1, 2],      Y = [[1.0, 0.0, 2.0],
                      [0.5, 1.0, 0.0]]
     [3, 4],
     [5, 6]]
out(ij,jk->ik) = [[2.0, 2.0, 2.0],
                  [5.0, 4.0, 6.0],
                  [8.0, 6.0, 10.0]]
```

以 $\text{out}[0,2]$ 为例：$j$ 是被消掉的收缩字母，$\text{out}[0,2]=\sum_j X[0,j]\,Y[j,2]=1\cdot 2.0+2\cdot 0.0=2.0$。

## 约束

- 支持 1–3 个操作数、每个操作数至多 4 根轴；输入是 float32 随机张量，与 `torch.einsum` 对拍容差 atol=1e-5、rtol=1e-5
- 禁止调用 `torch.einsum` / `torch.tensordot`；判题用非法 spec 探针逐个断言 ValueError
