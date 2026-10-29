# 详解：einsum 表达式求值器（标注串 → 乘加归约）

## 出处

Einstein 记号在 numpy 里由 `np.einsum`（2011 年引入）实现，PyTorch 提供同名 `torch.einsum`。面试题常以「不用 einsum，手写 `QK^T` 的 batch 版本」出现，本题把口径反过来：给定标注串，要求解出轴对齐与收缩集合，从而一次性覆盖 matmul / 转置 / 外积 / 按轴求和。

## 完整推导与下标约定

设 spec 拆成输入标注 $L_1,\dots,L_n$ 与输出标注 $M$。定义

- 收缩字母集 $\mathcal{C} = \bigcup_k L_k \setminus M$（按首次出现顺序排成一个列表）；
- 输出字母集 $M$（题面保证不重复且都出现在输入里）。

取统一轴序 `order = C ++ M`。对第 $k$ 个操作数，它在 order 中拥有的字母是
$\text{kept}_k = [c \in \text{order} \mid c \in L_k]$，对应的原始维下标
`perm_k = [L_k.index(c) for c in kept_k]`。因为 order 覆盖 $L_k$ 的每个字母，
`perm_k` 恰是 $\{0,\dots,\dim(T_k)-1\}$ 的一个排列，于是

$$T'_k = \text{permute}(T_k;\text{perm}_k)\ \text{reshape}\ \text{shape}[i] = \begin{cases} |L_k\text{ 中该字母的轴长}|, & c \in L_k\\ 1, & c \notin L_k \end{cases}$$

之后逐元素相乘靠广播自动完成「外积 / 点积 / batch 对齐」三件事：

$$P[\text{order}] = \prod_k T'_k[\text{order}], \qquad \text{out}[M] = \sum_{\mathcal{C}} P$$

由于 order 把收缩字母放在**最前面**、输出字母按 $M$ 的顺序放在后面，求和掉前
$|\mathcal{C}|$ 根轴以后剩下的轴天然就是 $M$ 的顺序，**不需要再做一次 permute**——
这是本题最容易多写一步的地方。

以 `"ij,jk->ik"` 为例：$M=\{i,k\}$、$\mathcal{C}=\{j\}$、order $=[j,i,k]$；
$X$ 的 kept $=[i,j]$ → perm $=[1,0]$ → shape $(2,3,1)$；$Y$ 的 kept $=[j,k]$ → perm $=[0,1]$ → shape $(2,1,3)$；相乘得 $(2,3,3)$，沿轴 0 求和 → $(3,3)$ ✓。

## 核心与易错点

1. **输出轴顺序**：`"ij->ji"` 与 `"ij->ij"` 的差别只在 order 的尾部顺序，若实现里
   「先按字母表排序再归约」就会把转置写成恒等。
2. **缺失字母要补长度 1，而不是删除**：`"i,j->ij"` 中第一个操作数没有 $j$ 轴，
   必须 reshape 成 $(3,1)$、第二个成 $(1,4)$，广播出 $(3,4)$。直接 `expand` 再乘也行，
   但 `reshape` 在 permute 后的非连续视图上会隐式 copy，语义仍然正确。
3. **全收缩返回 0 维**：`"i,i->"` 的 `out.dim() == 0`，写成返回 Python float 会挂形状断言。
4. **对角/trace 不支持**：`"ii->i"` 需要取对角（`torch.diagonal`），题面把它划到子集之外，
   要求抛 ValueError；把它当成合法输入的实现在这里必然翻车。
5. **精度**：在 float64 下乘加、末尾 cast 回 float32，与 `torch.einsum` 的 float32 累加
   在 $k\le 6$ 的量级上差异远小于 1e-5，容差安全。

## 与邻近题的区别

- `batched-gemm`：只交 matmul 的广播语义（`(...,n,m)@(...,m,p)`），轴是隐式的；本题的轴
  由字符串命名，可以任意置换与收缩。
- `transpose-view`/`concat-stack-split`：纯形状操作，没有乘加。
- `scaled-dot-product-attention`、`grouped-query-attention`：einsum 在这些题里是**手段**，
  本题里是**对象**。

## 对拍口径与复杂度

- 判题 oracle 直接是 `torch.einsum`（受限子集内逐点一致），另含结构断言（形状、dtype、0 维）
  与七个非法 spec 的 ValueError 探针。
- 复杂度：乘积张量的规模是 $\prod_{c\in\text{order}}|c|$，比逐收缩的实现更吃内存，
  但判题规模 $\le 6$ 根轴、轴长 $\le 6$，实测单题 CPU 毫秒级。
