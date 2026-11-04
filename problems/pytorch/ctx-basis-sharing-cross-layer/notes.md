# 详解：Basis Sharing 跨层共享基低秩分解

## 出处

论文 Basis Sharing（arXiv:2410.03765，ICLR 2025，Jingcun Wang 等，TU Darmstadt + UNL，官方仓 `TUDa-HWAI/Basis_Sharing`）。动机：SVD 压缩逐层独立进行时，不同层学出的基向量高度相关——跨层共享这组基可以把「每层一份完整基」的开销摊平成「全组一份基 + 每层窄系数」。论文系统实验了**哪些权重矩阵值得共享**（q/k/v/o、gate/up/down 各配独立 basis 池与分组）与**分组选择**，大压缩比下超过 SVD-LLM 等白化 SVD 基线。校准预处理直接复用 SVD-LLM 的 Cholesky 白化（calib.py 注释明说 code from SVD-LLM）。

**查重边界**：单层「Cholesky 白化 + 截断 SVD」已被 `sparsity-svdllm-whitened-truncation`（=773）钉死；本题增量是**跨层拼接联合分解 + 共享基 + 系数列段拆分 + 最小二乘重解 + 同输入校准复用**。KV 侧跨层 SVD 共享另有 `kv-xkv-layer-svd-share`（对象是 KV cache 投影维度，不是权重矩阵）。

## 官方实现的逐行翻译

```python
# group.py: Group._init_basis_coefficient（llama2 路径）
w = [model[layer].get_submodule(name).weight.data.T for layer in member]
w = torch.cat(w, -1).double()               # (in, m*out) 宽矩阵
w = s @ w                                   # SVD-LLM 白化（S = chol(H).T）
u, sigma, v = torch.svd(w)
self.basis = invs @ u @ torch.diag(sigma)   # (in, in) 全量，截断在 change_basis
self.coefficient = v.T                      # (in, m*out)，列段 = 层

# change_basis / change_coefficient: 截断到 num_basis
basis_name_weight = self.basis[:, :num_basis]      # (in, k)
weight = co[layer][name][:num_basis, :]            # (k, out_l)

# update_model: basis 变化后的系数闭式解
u = basis[:, :]                                    # (in, k) 已截断
inv = torch.inverse(u.T @ xtx @ u)
vt = w.T @ xtx @ u @ inv                           # (m*out, k)
v = vt.T                                           # (k, m*out)，列段 = 层
```

三个关键口径：

1. **白化作用在宽矩阵上**：`w = s @ w_cat` 一次性乘（与逐层乘再拼接由分配律数值恒等——等价变异剔除记录，勿再钉）；\(S = \mathrm{chol}(H)^\top\) 上三角，\(H = \Sigma x^\top x\) 校准 Hessian。
2. **系数的两条路径**：初建 \(C = v^\top\) 列段（SVD 右因子的转置，列段 = 层）；更新 \(C = (W^\top H B)(B^\top H B)^{-1}\)（先 (m·out, k) 再转置，列段 = 层）。**\(H = S^\top S\) 时两者恒等**（LS 目标经 \(H\) 加权展开后正是白化域 SVD 的最优低秩近似）——update 阶段的存在意义是 basis 被继续训练（lora.py）或替换后重解系数。
3. **同输入投影共享校准**：q/k/v 都吃 hidden_states，H 相同（q/v 复用 k 的校准文件）；gate/up 同理（gate 复用 up）。o/down 独立。映射在 `get_calib_data` 与 `update_model` 的 basis_name 分支两处出现，口径一致。

**前向链**（llama.py）：`y = q_proj(q_basis(x))`，即 \(x \to B \to C_l\)；等价于用重建权重 \(\hat W_l = (BC_l)^\top\) 的普通线性层。MLP 侧 `down(act(gate_basis(x)·系数) * up_basis(x)·系数)` 三投影各自共享。

## 易错点

1. **拼接轴与转置**：llama 权重先转置再沿 -1 拼（列段 = 层）；不转置或 axis=0 都破坏「列段=层」的系数切分前提（m-notrans/m-cat-axis 钉）。
2. **共享基三件套**：invs（去白化）、diag(σ)（能量加权）、前 k 列（截断）缺一不可（m-basis-noinv/m-basis-nosigma 钉）；取尾 k 列而非前 k 列破坏能量排序语义（m-basis-bottomk 钉）。
3. **系数切段偏移**：段起点是 \(i \cdot out\) 不是 \(i\)（m-coef-offset 钉）；截断取前 k **行**（vᵀ 的行 = 成分序），取前 k 列是转置混淆（m-coef-rowswap 钉）；漏截断则参数量账崩（m-coef-nok 钉）。
4. **LS 闭式解结构**：Gram 逆是 \((B^\top H B)^{-1}\) 不是 \((B^\top B)^{-1}\)（m-ls-noinv 钉）；H 加权在 \(W^\top H B\) 处（m-ls-noH 钉）；转置时机在解完之后（m-ls-rowsplit 钉）。
5. **重建/前向因子序**：\(\hat W = (BC)^\top\) 不是 \(CB\)（m-recon-order 钉）；前向 \(x\,B\,C_l\) 顺序错则维度/数值双崩（m-fwd-order 钉）。
6. **校准映射**：q/v→k、gate→up 两条复用（m-calib-qv/m-calib-gate 钉）。
7. **等价变异剔除记录**：白化「逐层乘再拼」与「拼后整体乘」由分配律恒等（m-white-perlayer 实测 SURVIVED，勿再钉）。

## 对拍口径

锚点由沙箱 dump（numpy float64，seed 57）：in=8/out=6/m=3/k=4，H 由 \(X^\top X\) 构造 SPD，S = chol(H)ᵀ 上三角，thin SVD 因子显式入参（不现跑分解，符号冻结）。四路互证——① 满秩重建：\(B_{full} C_{full} = W_{cat}\)（invs 可逆下精确）；② 满秩 LS 解 = 全量 \(v^\top\)；③ **两阶段恒等**：SVD basis 下 LS 重解 ≡ 初建系数（\(H = S^\top S\) 恒等的数值验证）；④ 扰动 basis 下 LS 满足正规方程残差 ≈ 0 且与初建可观测不同。前向 ≡ \(x\hat W^\top\) 同源互证。19 变异 18 CAUGHT + 1 分配律等价剔除。

## 复杂度

分解成本 \(O(in^2 \cdot m \cdot out)\)（离线一次）；压缩后参数 \(in \cdot k + m \cdot k \cdot out\)，对 \(m\) 层组省 \((m{-}1)(in - k) \cdot out\)；前向每层 \(O(BN(in\cdot k + k\cdot out))\)，比原 \(O(BN \cdot in \cdot out)\) 低 \(k / \min(in,out)\) 倍。共享基的存储摊还随组深线性放大收益。
