# MoE：分组矩阵乘的 offsets 切分与前反向

## 任务

实现 `solve`：专家权重按 $E$ 个组堆在一个 $[E,K,N]$ 的张量里，token 已经按专家 id 排好序堆在 $[S,K]$ 里，两边只靠一条**前缀和式的 `offs`** 对齐。算出分组矩阵乘的输出、对输入的梯度、对权重的梯度，外加每组到底吃了多少行。

这是 MoE 高性能执行路径的公共底座：`torch.nn.functional.grouped_mm`、`torch._grouped_mm`、各类 Triton/CUTLASS grouped GEMM 都吃同一套「按 expert 排序 + offsets 描述 ragged 边界」的约定，而它们不可用时的 fallback 就是一个「循环 $E$ 次、每次切一段做 `torch.mm`」的朴素实现。面试考这一段，考的不是数学而是**边界口径**：`offs` 是含端点还是不含、空组要不要跳过、排序后没填满的部分算谁的、前反向的切分是不是严格对称。本题只到纯函数为止：不涉及 all-to-all、不涉及 permute/unpermute 的实际搬运、不建计算图也不需要 autograd。

## 输入

- `x`：CPU float64 `[S,K]`，非空 2-D、有限。行按专家 id 排序，第 $e$ 组占连续的一段行。
- `weight`：CPU float64 `[E,K,N]`，非空 3-D、有限。`weight[e]` 是第 $e$ 个专家的线性层权重。
- `offs`：CPU int32 或 int64 `[E]`，非空 1-D。**每个元素是该组结束处的行下标**（前缀和、含左不含右），因此必须非降、每个元素 $\ge 0$、且 $\mathit{offs}[E-1]\le S$。
- `grad_output`：CPU float64 `[S,N]`，非空 2-D、有限。上游传下来的梯度，行序与 `x` 一致。

维度对齐要求 `x.size(1) == weight.size(1)`、`weight.size(2) == grad_output.size(1)`、`grad_output.size(0) == x.size(0)`、`offs.size(0) == weight.size(0)`。接受非连续张量，不修改输入。

## 精确规则

记 $\mathit{offs}[-1]\equiv 0$（书写方便，不是输入元素）。第 $e$ 组覆盖的行是**左含右开**区间

$$
\mathit{start}_e=\begin{cases}0,&e=0\\ \mathit{offs}[e-1],&e>0\end{cases}\qquad
\mathit{end}_e=\mathit{offs}[e]\qquad \mathit{count}_e=\mathit{end}_e-\mathit{start}_e .
$$

四条口径，全部照字面执行：

- **前向**：$\mathit{out}[\mathit{start}_e:\mathit{end}_e]=x[\mathit{start}_e:\mathit{end}_e]\ @\ \mathit{weight}[e]$。
- **空组**（$\mathit{count}_e=0$）整段跳过：不产生任何数值，$\mathit{grad\_weight}[e]$ 保持全 0。就空组本身而言「乘一下」和「跳过」给出完全相同的结果（0 行的矩阵乘不会读权重），真正会被判出来的是把**非空组**的 `weight` 下标弄错，或者弄错了区间划分。
- **未分配的尾部行**：$[\mathit{offs}[E-1],\,S)$ 这些行不属于任何专家，它们的前向输出行和输入梯度行**必须是 0**，不参与任何 $\mathit{grad\_weight}$ 的贡献。
- **反向**（与切分严格对称）：
$$
\mathit{grad\_input}[\mathit{start}_e:\mathit{end}_e]=\mathit{grad\_output}[\mathit{start}_e:\mathit{end}_e]\ @\ \mathit{weight}[e]^{\top},\qquad
\mathit{grad\_weight}[e]=x[\mathit{start}_e:\mathit{end}_e]^{\top}\ @\ \mathit{grad\_output}[\mathit{start}_e:\mathit{end}_e].
$$

`counts[e]` 就是 $\mathit{count}_e$，由 `offs` 差分得到，不需要真的数行。

非法输入（dtype/形状/设备/非有限/`offs` 非整数或降序或越界/维度不匹配）抛 `ValueError`。

## 返回

`(out, grad_input, grad_weight, counts)`：

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `out` | float64 `[S,N]` | 分组矩阵乘结果，尾部行为 0 |
| `grad_input` | float64 `[S,K]` | 对 `x` 的梯度，尾部行为 0 |
| `grad_weight` | float64 `[E,K,N]` | 对 `weight` 的梯度，空组为 0 |
| `counts` | int64 `[E]` | 每组吃到的行数，和为 $\mathit{offs}[E-1]$ |

浮点用 `rtol=atol=2e-11`，`counts` 逐位比较。

## 示例

$S=6$、$E=3$、$K=2$、$N=2$：

```python
x   = [[1, 2], [0, -1], [2, 1], [1, 0], [-1, 2], [3, 3]]   # S=6, K=2
w0  = [[1, 0], [2, 1]]   w1 = [[1, 1], [1, 1]]   w2 = [[0, 2], [-1, 1]]   # E=3, N=2
offs = [2, 2, 5]
grad = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [7, 7]]

# counts       [2, 0, 3]
# out          [[5, 2], [-2, -1], [-1, 5], [0, 2], [-2, 0], [0, 0]]
# grad_input   [[1, 2], [0, 1], [2, 0], [0, -2], [4, 2], [0, 0]]
# grad_weight  e0: [[1, 0], [2, -1]]   e1: [[0, 0], [0, 0]]   e2: [[4, 0], [1, 5]]
```

逐组核对：

- **e0** 吃行 $[0,2)$：$[1,2]\cdot[[1,0],[2,1]]=(1+4,\ 0+2)=(5,2)$；$[0,-1]$ 得 $(-2,-1)$。梯度侧 $\mathit{grad\_weight}[0]=x_{0:2}^{\top}\mathit{grad}_{0:2}$：$(0,0)$ 元 $=1\cdot1+0\cdot0=1$，$(1,1)$ 元 $=2\cdot0+(-1)\cdot1=-1$。
- **e1** $\mathit{offs}[1]=\mathit{offs}[0]=2$ → 空组，`counts[1]=0`，$\mathit{grad\_weight}[1]$ 全 0。注意 $w_1$ 在这条数据里完全没被读到，把它换成任意值都不会改变四个返回项。
- **e2** 吃行 $[2,5)$（3 行）：$[2,1]\cdot w_2=(-1,5)$、$[1,0]\cdot w_2=(0,2)$、$[-1,2]\cdot w_2=(-2,0)$。$\mathit{grad\_input}$ 用 $w_2^{\top}$：第 2 行 $(1,1)\cdot w_2^{\top}=(1\cdot0+1\cdot(-1),\ 1\cdot2+1\cdot1)=(2,0)$。
- **行 5** 在 $[\mathit{offs}[2],S)=[5,6)$ 里，没有专家认领：`out` 第 5 行是 $[0,0]$，`grad_input` 第 5 行是 $[0,0]$，它的梯度 $[7,7]$ 也不进任何 $\mathit{grad\_weight}$。

所有数字都是整数乘加，可以手算到底。
