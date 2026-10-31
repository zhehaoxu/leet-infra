# 量化：GEAR 量化误差的低秩幂迭代补偿

## 题目描述

GEAR（arXiv 2403.05527）不改造量化器本身，而是把量化留下的误差再拆一次：一个稀疏项吸收离群元、一个低秩项吸收剩下的量化残差，KV cache 于是存成「低比特码字 + 少量离群元 + 一对低秩因子」。本题把这条链路的三段依次实现出来。

1. **离群元（GEAR 的 S）**：把 `(B, H, L, D)` 沿 head 与 head_dim 合并成 token 行 `(B, L, C=H*D)`，每行取 `n_out` 个最小值和 `n_out` 个最大值，先把它们换成**整行的原始均值**（离群元自己也算在内）送去量化，量化完再把离群元的**原始值**写回，于是这些位置的残差恒为 0；
2. **分组非对称量化**：每个 token 行沿连续 `group_size` 切组，scale \(=(mx-mn)/(2^{\text{bits}}-1)\)、code \(=\mathrm{round}(\max((x-mn)/\text{scale},\,0))\)、反量化 \(\hat x=\text{code}\cdot\text{scale}+mn\)；
3. **低秩补偿（GEAR 的 L）**：对每个 \((b,h)\) 的残差 \(A\)（形状 \(L\times D\)）跑 `loop` 轮幂迭代，两次 QR 都只发生在**最后一轮**；收尾时左右基 \(Q,P\) 都已正交化且 \(P=A^{\top}Q\)，`low_rank` \(=Q P^{\top}=Q Q^{\top}A\) 是 \(A\) 到 \(\operatorname{span}(Q)\) 的**正交投影**，所以补偿只会让误差变小、不会变大；`output` \(=\) `quantized` \(+\) `low_rank`。这里的名额是整行的、替换值含离群元、级数是 \(2^{\text{bits}}-1\) 而非 `bits**2 - 1`，都是判分口径。

## 任务

实现 `solve(kv, bits, group_size, sparsity, rank, loop, p_base, q_base)`：

- `kv`：float32/float64，`(n_batch, n_head, seq_len, head_dim)`，即上游 GEARLM 的 K/V 布局；
- `bits`、`group_size`、`rank`、`loop`：正整数；`group_size` 须整除 `n_head*head_dim`，`rank <= min(seq_len, head_dim)`；
- `sparsity`：落在 \([0,1)\) 内的实数，整行的离群元占比；
- `p_base`：`(n_batch, n_head, head_dim, rank)`，右侧基的初值；`q_base`：`(n_batch, n_head, seq_len, rank)`，左侧基的初值；两者与 `kv` 同 dtype、只读。

内部一律 float64，返回按 `kv` 的 dtype，六个键的形状全部与 `kv` 相同：

- `outlier_mask`：bool，被判为离群元的位置；
- `sparse_filled`：两端极值被行均值替换之后的张量；
- `quantized`：分组量化、并把原始极值写回之后的张量；
- `residual`：`kv - quantized`，离群位置恒为 0；
- `low_rank`：残差的秩-`rank` 幂迭代近似；
- `output`：`quantized + low_rank`。

非法输入抛 `ValueError`：`kv` 不是四维或不是 float32/float64；四个整数参数不是 int（`bool` 也不算）或小于 1；`sparsity` 是 `bool` 或不在 \([0,1)\)；`group_size` 不整除 `n_head*head_dim`；`rank` 超过 `min(seq_len, head_dim)`；两个基的形状不对或 dtype 与 `kv` 不一致。

## 示例

```
kv[0, 0] = [[-4, 1, 2, 3], [-4, 1, 2, 3], [-4, 1, 2, 3]]      # head0
kv[0, 1] = [[ 4, 5, 6, 7], [ 4, 5, 6, 8], [ 4, 5, 6, 9]]      # head1
bits = 2   group_size = 4   sparsity = 0.25   rank = 1   loop = 1
p_base = ones(1, 2, 4, 1)      q_base = zeros(1, 2, 3, 1)

# 1) 整行名额：head 合并后每行 8 个通道，两端各 1 个
n_out = int(int(1*2*3*4*0.25)/1/3/2) = int(6/6) = 1
row0 = [-4, 1, 2, 3 | 4, 5, 6, 7]   mean = 24/8 = 3           # 均值含 -4 和 7
row1 = [-4, 1, 2, 3 | 4, 5, 6, 8]   mean = 25/8 = 3.125
row2 = [-4, 1, 2, 3 | 4, 5, 6, 9]   mean = 26/8 = 3.25        # 每行换掉 -4 与本行的 7/8/9
sparse_filled[head0] = [[3, 1, 2, 3], [3.125, 1, 2, 3], [3.25, 1, 2, 3]]
sparse_filled[head1] = [[4, 5, 6, 3], [4, 5, 6, 3.125], [4, 5, 6, 3.25]]

# 2) 分组量化（2**bits - 1 = 3 个间隔），看 row1 的 head1 组 [4, 5, 6, 3.125]
mn = 25/8   mx = 6   scale = (6 - 25/8)/3 = 23/24
4     -> (4 - 25/8)/scale = 21/23 ≈ 0.9130 -> code 1 -> 25/8 + 23/24 = 49/12 ≈ 4.0833
5     -> (5 - 25/8)/scale = 45/23 ≈ 1.9565 -> code 2 -> 25/8 + 23/12 = 121/24 ≈ 5.0417
6     -> (6 - 25/8)/scale = 3             -> code 3 -> 6
3.125 -> 0                                -> code 0 -> 25/8，随后原始离群元 8 写回组末
quantized[head1, row1] = [49/12, 121/24, 6, 8]   residual[head1, row1] = [-1/12, -1/24, 0, 0]

# 3) 低秩补偿：每个 (b, h) 的残差 A 是一张 3x4 矩阵
residual[head1] = [[0, 0, 0, 0], [-1/12, -1/24, 0, 0], [-1/6, -1/12, 0, 0]]     # 第 3 行 = 2× 第 2 行，秩 1
residual[head0] = [[0, 0, -1/3, 0], [0, 0, 7/24, -1/8], [0, 0, 1/4, -1/4]]      # 两列无关，秩 2
qr(p_base).Q 只给一个方向 (1, 1, 1, 1)/2，rank=1 的投影补不满秩 2：
low_rank[head0] = [[0, 0, -23/60, 1/20], [0, 0, 23/120, -1/40], [0, 0, 0, 0]]
output[head1] == kv[head1]（逐元素还原）    ||kv - output|| = 0.3873 < ||kv - quantized|| = 0.6166
```

head1 的第一行整组刚好踩在格点上（scale 是 1），残差只剩第 2、3 行的前两列且两行成比例——数学上秩 1，秩-1 投影全额补回、逐元素还原成输入；head0 的残差占第 3、4 两列且互不成比例，秩 2，秩-1 只能吃掉一多半。

## 约束

- **名额按合并后的整行给，且两次向下取整**：`n_out = int(int(B*H*L*D*sparsity)/B/L/2)`，外层先把元素总数截成整数，除式里只有 `B`、`L`、`2`（没有 `H`）。示例形状下 `sparsity` 取 0.24/0.25/0.5 时 `n_out` 分别是 0/1/2。
- **替换值是含离群元的整行原始均值**：不是截尾均值。`[-100, 0, 0, 0, 0, 0, 0, 100]` 这一行替换后整行都是 0；两端分两次写回，同一个下标同时入选时结果不变。
- **同分取更小的下标（两端都是）**：实现用 stable argsort，等价于按 `(值, 下标)` 排序后取前 `n_out`；上游 `torch.topk` 的同分顺序是未定义的。
- **级数是 `2**bits - 1` 个间隔**：code 落在 `[0, 2**bits - 1]`。不是 `bits**2 - 1`——后者是上游笔误，只在 `bits ∈ {2, 4}` 时与前者巧合相等。
- **常数组反量化成组 min**：`scale == 0` 时不做除法（分母兜底成 1），code 恒 0，结果为 `mn`，不是 NaN。
- **`q_base` 白给，`p_base` 定子空间**：`loop >= 1` 时 `q_base` 在第一轮就被 `A @ P` 整体覆盖，改成任何值 `low_rank` 逐元素不变；`p_base` 只通过最后一轮 QR 后的列张成子空间，整体取负或缩放都不影响 `low_rank`。
- **补偿只可能缩小误差**：`low_rank` 是正交投影，故 `||kv - output|| <= ||kv - quantized||`，且每个 \((b,h)\) 的 `low_rank` 秩不超过 `rank`。
- **dtype 与副作用**：内部 float64；`outlier_mask` 恒为 bool，其余五个键的 dtype 跟随 `kv`；三个输入张量不得就地修改，非连续视图的结果与连续副本一致。
