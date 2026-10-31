# MoE：分组执行的排序计划与还原合并

## 任务

实现 `solve`：把「每个 token 选了哪几个专家」这张表编译成分组矩阵乘（grouped mm）真正需要的索引计划，算完之后再把它还原、按 token 合并成一个输出。

这是 MoE 专家侧 `grouped_mm` / `batched_mm` 实现的受限数值模型：排序、偏移表、gather/逆排列、哨兵行的处理、合并方式都由题面显式规定。专家的层内计算被抽象成一次逐元素对角乘（`hidden[t] * diag[e]`），因为本题考的是**槽位代数**而不是矩阵乘；不涉及反向传播、CUDA graph 或真正的 kernel 调度。

## 输入

- `hidden`：CPU float64 `[T,D]`，非空。
- `expert_ids`：CPU int64 `[T,k]`；取值合法范围是 $0,\dots,E$，其中 $E$ **本身就是哨兵值**（该槽位被丢弃，或专家不在本卡）。
- `routing_weights`：CPU float64 `[T,k]`，与 `expert_ids` 同形状；不做重新归一化。
- `diag`：CPU float64 `[E,D]`，第二个维度必须与 `hidden` 一致；`E = diag.shape[0]`，题目不单独传专家数。
- `T,k,E = 1..8`，`D = 1..4`。

允许非连续张量，不修改任何输入。id 越界（负数或大于 $E$）、形状不匹配、dtype 不符、不在 CPU、维度不对、有空维都抛出 `ValueError`。

## 精确规则

槽位按 token-major 编号：第 $s = t\,k + j$ 个槽位就是 token $t$ 的第 $j$ 个选择，共 $S = Tk$ 个槽位。

**排序**：按二元组（专家编号，槽位编号）升序且**稳定**，得到 `perm`；哨兵编号最大，因此**自然排在尾部**，有效行恰好是前 $\mathit{valid}$ 行：

$$
(i<j)\Rightarrow\bigl(e_{\mathit{perm}[i]}<e_{\mathit{perm}[j]}\ \text{or}\ (e_{\mathit{perm}[i]}=e_{\mathit{perm}[j]}\ \text{and}\ \mathit{perm}[i]<\mathit{perm}[j])\bigr),\qquad
\mathit{token\_rows}[i]=\bigl\lfloor \mathit{perm}[i]/k \bigr\rfloor .
$$

`token_rows[i]` 说明排序后第 $i$ 行该 gather 哪个 token 的 hidden；下文记 $\mathit{sorted\_experts}[i]=e_{\mathit{perm}[i]}$，它是推导量、不需要返回。

**偏移表**：`counts[e]` 只统计 $0\le e<E$ 的槽位（哨兵不计入），`offsets` 是 `counts` 的**闭前缀和**，长度 $E$：

$$
\mathit{counts}_e=\#\{s: e_s=e\},\qquad \mathit{offsets}_e=\sum_{e'\le e}\mathit{counts}_{e'},\qquad \mathit{offsets}_{E-1}=\mathit{valid}=S-\#\{s:e_s=E\}.
$$

**分组输出**：令 $\hat e=\min(\mathit{sorted\_experts}[i],\,E-1)$（先 clamp 保证索引在界内），

$$
\mathit{grouped}[i]=\begin{cases}
h_{\mathit{token\_rows}[i]}\odot \mathit{diag}_{\hat e}, & \mathit{sorted\_experts}[i]<E,\\[2pt]
\mathbf{0}, & \text{sentinel} .
\end{cases}
$$

哨兵行必须整行为 0——不管 clamp 之后取到的是哪个专家。

**还原与合并**：`inv_perm` 满足 $\mathit{inv\_perm}[\mathit{perm}[i]]=i$，权重按排序后的顺序乘上去，再还原成 token-major 并对 $k$ 求和：

$$
\mathit{out}_t=\sum_{j=0}^{k-1}\bigl(\mathit{grouped}\odot w_{\mathit{perm}}\bigr)\bigl[\mathit{inv\_perm}[\,tk+j\,]\bigr].
$$

合并用的是「unsort → reshape → sum」，不是 `index_add_`：前者的累加顺序就是 $j=0,\dots,k-1$ 的固定顺序。

## 返回

返回七元 `tuple`，$S=Tk$：

```text
(out, grouped, perm, inv_perm, token_rows, counts, offsets)
```

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `out` | float64 `[T,D]` | 合并后的 token 输出 |
| `grouped` | float64 `[S,D]` | 排序后的逐专家输出，哨兵行整行为 0 |
| `perm` | int64 `[S]` | 排序后第 $i$ 行来自哪个原槽位 |
| `inv_perm` | int64 `[S]` | 槽位 $s$ 在排序后落在第几行 |
| `token_rows` | int64 `[S]` | 排序后第 $i$ 行要 gather 的 token |
| `counts` | int64 `[E]` | 每个专家的槽位数（不含哨兵） |
| `offsets` | int64 `[E]` | `counts` 的闭前缀和 |

浮点结果按 `rtol=2e-11, atol=2e-11` 比较，索引张量逐位相等。

## 示例

```python
hidden = [[1.0, 2.0],
          [3.0, -1.0]]
expert_ids = [[1, 3],
              [0, 1]]          # E = 3, so id 3 is the sentinel
routing_weights = [[0.5, 0.2],
                   [0.25, 1.0]]
diag = [[0.5, -1.0],
        [2.0, 0.0],
        [-1.0, 3.0]]

# perm        = [2, 0, 3, 1]
# token_rows  = [1, 0, 1, 0]
# inv_perm    = [1, 3, 0, 2]
# counts      = [1, 2, 0]
# offsets     = [1, 3, 3]
# grouped     = [[1.5, 1.0], [2.0, 0.0], [6.0, 0.0], [0.0, 0.0]]
# out         = [[1.0, 0.0], [6.375, 0.25]]
```

四个槽位的专家编号依次是 `[1, 3, 0, 1]`。按（专家，槽位）升序排：槽位 2（专家 0）、槽位 0（专家 1）、槽位 3（专家 1）、槽位 1（专家 3=哨兵），于是 `perm = [2,0,3,1]`。`token_rows = perm // 2 = [1,0,1,0]`。

`counts`：专家 0 一个、专家 1 两个、专家 2 零个；哨兵不计数。`offsets = [1,3,3]`，有效行是前 3 行。

`grouped` 三行分别是 `hidden[1]*diag[0] = [1.5, 1.0]`、`hidden[0]*diag[1] = [2.0, 0.0]`、`hidden[1]*diag[1] = [6.0, 0.0]`；第四行是哨兵，整行 0。

乘上排序后的权重 `[0.25, 0.5, 1.0, 0.2]`（即 `flat_w[perm]`）得 `[[0.375,0.25],[1.0,0.0],[6.0,0.0],[0,0]]`；用 `inv_perm` 还原成 token-major 后，token 0 的两项是 `[1.0,0.0] + [0,0] = [1.0,0.0]`，token 1 是 `[0.375,0.25] + [6.0,0.0] = [6.375,0.25]`。直接按 token 验算也一致：token 0 只有专家 1 这一路有效，`0.5 * [2,0] = [1,0]`；哨兵那一路即使权重是 0.2 也必须贡献 0。
