# 详解：GLoRA 权重合并与精确逆解

## 出处

peft `tuners/glora.py`（GLoRA：Generalized LoRA, arXiv:2410.01485）。核心是把增量拆成「基座权重的逐元素缩放」与「加性低秩项」，因而一条适配器只有两条路径：

- \(A\) 路径（`config_A_B`）：`lora` 为 \((d_\text{out},r)@(r,d_\text{in})\)，`vector` 为 \((d_\text{out},1)\) 列向量，`constant` 为标量，`empty` 为常数 0；
- \(B\) 路径同理；
- 偏置侧：\(C\) 走 \((d_\text{in},1)\) 低秩/列向量，\(D,E\) 走 \((d_\text{out},)\) 向量/标量。

本题固定 \(A,B\) 两条路径都用 `lora`、偏置侧用向量形式，这是 GLoRA 论文与 PEFT 默认配置的主干。

## 合并公式（`GloraLayer.merge`）

```
new_weight = w0.clone()
for each adapter i:
    new_weight += w0 * A_i + B_i
    new_bias   += b0 * D_i + E_i + W0 @ C_i
```

三个易错点：

1. **累加器必须独立于 `w0`**。若直接在已合并权重上做 `new_weight += new_weight * A + B`，第二个适配器就会乘到 \(W_0+\Delta_1\) 上，产生交叉项。peft 这里专门留了注释说明：多适配器的每一项都要以原始 \(W_0\) 计算。
2. **偏置的 \(C\) 项是 \(W_0C\)（矩阵乘），不是 \(b_0\odot C\)**。\(C\) 作用在**输入维**上，先与权重相乘把输入空间的信息搬进输出空间，才得到 \((d_\text{out},)\)。
3. **逆解的 \(W_0\) 要先解出来**。`unmerge` 是 \(W_0=(W'-\sum B)/(1+\sum A)\)，然后才用它去算 \(\sum W_0C\)，即偏置反解里的 \(C\) 项用的是**恢复后的**基座权重。

## 为什么"未合并前向"能对上

`GloraLinear.forward` 未合并时逐适配器算 `F.linear(x, W0* (1+A_i)+B_i, b0*(1+D_i)+E_i+W0@C_i)`，最后 `result -= n * base_result`（先从 base 起算再加回 n 份，等价于减 \((n-1)\) 份）。展开后权重恰好是 \(W_0+\sum_i(W_0A_i+B_i)\)，与 merge 完全一致——这就是 `check_unmerged_forward_matches_merged` 的对拍口径，也是 `-n*base` 这一行代码存在的理由。

## 对拍口径

- `check_example`：单适配器样例，同时钉死 \(W'\)、\(b'\)、\(y\) 三组数值（float32，容差 \(10^{-5}\)）。
- `_dense` 用显式秩一求和 \(\sum_k d_{\cdot k}\otimes u_{k\cdot}\) 展开 \(A_dA_u\)，与参考实现的 `@` 结构不同。
- `_unmerged_forward` 走"逐适配器 effective 权重"路线，与参考的"先合并再前向"互为独立实现。
- `check_zero_init_identity`：peft 把 `Xd` 初始化为全零，故 \(A=B=C=D=E=0\)、合并后严格等于原层——用 `torch.equal` 钉死（不是 `allclose`）。
- `check_bias_side_path`：只留 \(C\)，验证 \(b'-b_0=W_0C\) 且 \(W'=W_0\)。
- `check_multi_adapter_no_compounding`：先对准参考，再断言"顺序复合"的错误结果与之偏离，防止把累加写成滚动更新。

## 复杂度

合并是 \(O(nd_\text{out}d_\text{in})\) 的元素级累加加 \(n\) 次低秩矩阵乘；前向与普通 `Linear` 同阶，推理时没有任何适配器额外开销（这正是 merge 的意义）。
