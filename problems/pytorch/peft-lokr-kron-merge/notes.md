# PEFT：LoKr Kronecker 积权重合并——算法详解

## 1. 公式与 PEFT 口径

LoKr 出自 Yeh et al., arXiv:2108.07383（LyCORIS `lokr`），PEFT 在 `tuners/lokr/layer.py` 实现。Linear 形态：

- `factorization(in_dim)`/`factorization(out_dim)` 把维度拆成乘子（见《LoKr Kronecker 因子分解》题），得到形状四元组 \((a,b)\times(c,d)\)；
- full 形态 `lokr_w1: (a,b)`、`lokr_w2: (c,d)`（也可把任一因子再低秩化：`w1 = w1_a @ w1_b`）；
- `get_delta_weight` 调 `make_kron(w1, w2, self.scaling[adapter_name])`，而 `make_kron` 最后一行是 `torch.kron(w1, w2) * scale`，`scale = alpha / r`（`update_layer` 中与 LoRA/LoHa 同款）。

合并：\(W' = W_0 + s\,(W_1\otimes W_2)\)。

## 2. Kronecker 积的块布局（判题核心）

\[(W_1\otimes W_2)[\,i\,c+u,\; j\,d+v\,] = W_1[i,j]\cdot W_2[u,v]\]

记忆图：\(W_1\) 的每个元素"膨胀"成一个 \(W_2\) 形状的小块，块在行方向按 \(W_1\) 的行序排布、块内行按 \(W_2\) 的行序排布（列同理）。**两个因子不对称**：外层结构 × 内层纹理，因此 \(\operatorname{kron}(W_1,W_2)\neq\operatorname{kron}(W_2,W_1)\)（连形状都可能不同）。`check_matches_oracle` 用非方阵因子 \((2,3)\otimes(3,2)\) 检验布局方向——放反了行/列块序立刻对不上。

## 3. 参数账（面试常问）

- \(\Delta W\) 元素数：\((a\,c)(b\,d)\)；因子参数数：\(a\,b+c\,d\)。方阵情形 \(W_1,W_2\in\mathbb R^{k\times k}\)：参数 \(2k^2\)，表达 \((k^2)^2=k^4\) 的矩阵——\(k=16\) 时 512 参数 vs 65 536 元素，128× 压缩。
- 与 LoHa 对比：LoHa 走"满秩但逐元素门控"，LoKr 走"结构性稀疏（Kronecker 秩 1）"；两者都只需 \(O(d)\) 级参数（LoKr 甚至可以更低）。
- LyCORIS 风格初始化（`reset_adapter_parameters_lycoris_way`）置零 \(W_2\)：起点 \(\Delta W=0\)，对应 `check_zero_w2_degenerates_to_base`。
- \(r\) 在 full 形态里只承担缩放语义（\(s=\alpha/r\)）；只有当因子被低秩化（`use_w1/use_w2=False`）时 \(r\) 才决定旁路宽度。

## 4. 与本合集其他题的分工

- `peft-lokr-factorization`：本题的上游——维度怎么拆成 \((a,b),(c,d)\)。
- `peft-loha-hadamard`：另一条乘法路线（Hadamard vs Kronecker）。
- `peft-lora-merge`：加法合并 \(W_0+sBA\)。
