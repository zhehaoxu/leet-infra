# 详解：FRoD 旋转投影对角加稀疏增量

## 出处

FRoD（Frequency-domain Rotational Decomposition，AAAI 2026）已进 HF PEFT v0.20.0（`src/peft/tuners/frod/`）。设计动机：同类别（如全部 q_proj）的各层增量高度相关，逐层独立 LoRA 参数浪费——FRoD 跨层共享一个**旋转基** \(V\)，把每层增量表达为「对角 + 极稀疏非对角」。对角项等价于秩 1 缩放视角（每基方向一个系数），稀疏项 \(L\)（`sparse_rate` 控制密度）补足非对角修正。

## 官方实现的逐行翻译

**共享投影（`model.py::get_frod_projection`）**：

```python
w_cat = torch.cat([w for w in same_class_weights], dim=0)   # (Σout, in)
if w_cat.shape[0] < w_cat.shape[1]:        # 瘦阵: 行 < 列
    _, _, vh = torch.linalg.svd(w_cat.float(), full_matrices=False)
    V = vh                                  # 行正交
else:                                       # 宽阵: 行 ≥ 列
    q, r = torch.linalg.qr(w_cat.float().T, mode="reduced")   # q:(in,k), r:(k,Σout)
    invr = torch.linalg.inv(r.T @ r + alpha * torch.eye(...)) # 岭平均
    m = (invr @ r.T @ q.T).mean(...)        # 对称化 → eigh
    eigval, eigvec = torch.linalg.eigh(m)
    V = eigvec                              # 方阵 (in, in)
```

**稀疏 COO（`model.py`）**：

```python
all_idx = [(i, j) for i in range(n) for j in range(n) if i != j]  # 去对角
nnz = max(1, int(n * n * sparse_rate))
perm = torch.randperm(len(all_idx), generator=torch.Generator().manual_seed(prng_key))
s_indices = all_idx[perm[:nnz]]            # frod_s_indices, values 零初始化
```

**层内账（`layer.py::reset_frod_parameters` + `get_delta_weight`）**：

```python
wv = W @ V.T                               # (out, in)
if wv.shape[0] == wv.shape[1]:
    u = torch.linalg.solve(V, W.T).T       # U = W Vᵀ V⁻¹? → 官方 solve(V, Wᵀ)ᵀ: U diag(λ) = W Vᵀ
else:
    u = (torch.linalg.pinv(V) @ W.T).T
lambda_l = wv.norm(dim=0)                  # 列范数 (in,)
u = u / u.norm(dim=0, keepdim=True)        # 列归一化
s_values = zeros(nnz)
# get_delta_weight:
S = scatter(s_values, s_indices, (in, in))
delta_w = u @ (S + diag(lambda_l)) @ V.T - W   # 预校正减 W
```

**forward（`layer.py::forward` + `_sparse_activation_mm`）**：

```python
z = x @ V
z_S = _sparse_activation_mm(z, S)          # (S @ z.T).T = z @ S.T   ← 非对称 S 的转置细节！
z_L = z * lambda_l
out = base(x) + (z_S + z_L) @ u.T
```

## 三个数值口径（易错点）

1. **\(S\) 非对称，forward 是 \(zS^{\top}\)**：`_sparse_activation_mm` 的实现是 `S @ z.T` 再转置，等价于 `z @ S.T`。若想当然写 `z @ S`，与 merge 的 \(U(S+L)V^{\top}\) 转置不自洽（沙箱互证立刻暴露，m-fwd-notrans 钉）。对角分支 \(z*\lambda\) 无转置问题（对角矩阵对称）。
2. **\(S=0\) 时 \(\Delta W\equiv 0\) 精确成立**：\(U\mathrm{diag}(\lambda)V^{\top}=W\) 是构造恒等（\(U\lambda\) 就是 \(WV^{\top}\) 的极分解重排）。这是分解域参数化的「零初始化 = 零增量」语义，与 LoRA 的 B 零初始化同哲学。注意 \(L\ne0\) 时不再恒等——\(L\) 是训练后的稀疏残差。
3. **solve vs pinv**：方阵路径官方用 `torch.linalg.solve`；非方阵 W（out≠in）时 \(WV^{\top}\) 不可逆路径回退 `pinv`。病态 \(V\)（大条件数）下 pinv 的 rtol 会截断小奇异方向、数值可区分——测试用 cond≈\(10^8\) 的 fixture 钉住走的是 solve。

**λ 与 U 的归一化账**：\(\lambda\) 取 \(WV^{\top}\) 列范数、\(U\) 列归一化，保证 \(U\mathrm{diag}(\lambda)=WV^{\top}\) 仍精确；训练时 \(\lambda\)（对角）与 \(s\_values\)（稀疏）是可训练参数，\(U,V\) 冻结。

**float32 边界**：\(V\) 的分解、稀疏索引生成都在 float32（官方路径）；W/x/ΔW 外部账 float64。测试锚点按官方 dtype 路径 dump（V float32 值逐位复现）。

## 对拍口径

锚点由沙箱 dump：F1 宽阵路径（两层拼接 23 行>6 列，QR+岭+eigh）、F2 瘦阵路径（行<列，SVD 行正交）、F3 稀疏 COO（nnz 账 + 非对角 + seed 确定性）、F4 层内 solve/λ/归一化 + 重构恒等、F5 零 S 恒等 ΔW≡0、F6 非方阵 pinv 回退。互证：\(x\Delta W^{\top} = xV(S+L)^{\top}U^{\top}\)、S=0 精确零。

## 复杂度

共享投影一次性 \(O((\sum out)\cdot in^2)\)；每层前向 \(O(BN(in^2))\)（\(V\) 方阵乘主导）——比同秩 LoRA 略高（\(V\) 是 \(in\times in\) 而非 \(in\times r\)），但可训练参数只有 \(in + nnz\) 个对角+稀疏位，比 \(r(in+out)\) 小一个数量级。
