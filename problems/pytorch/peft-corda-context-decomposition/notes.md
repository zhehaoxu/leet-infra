# 详解：CorDA 上下文定向分解适配

## 出处

- 论文：**CorDA: Context-Oriented Decomposition Adaptation of Large Language Models**（arXiv:2406.05223，NeurIPS 2024，Yibo Yang 等，KAUST/HIT(Shenzhen)）。
- 官方实现：`iboing/CorDA`（`cordalib/decomposition.py` 的 `decompose_to_adapter` + `CorDA_adapter`；HF PEFT 另有 `examples/corda_finetuning`）。
- 关键差异（vs PiSSA/LoFTQ 一系）：CorDA 的分解方向由**协方差矩阵**决定——对 \(W\cdot C\) 做 SVD 再把 \(C^{-1}\) 乘回 \(V\)；PiSSA 直接对 \(W\) 做 SVD 取主成分。

## 官方实现的逐行翻译

**阻尼求逆**（decomposition.py，逐行）：

```python
covariance_matrix = linear.covariance_matrix.float()
damp = 0.01
while True:
    compensate = torch.diag(torch.ones(covariance_matrix.size(0))
        * torch.mean(torch.diag(covariance_matrix)) * damp)
    fix_covariance_matrix = covariance_matrix + compensate
    cov_inv = torch.linalg.inv(fix_covariance_matrix)
    inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(n))
    if inv_error < 0.05:
        break
    else:
        damp = damp * 2
```

```python
# 阻尼翻倍在判题域不可观测：所有合理输入的 inv_error 都在机器精度 ~4e-16 量级，
# 首步 damp=0.01 即收敛（tol=0.05 是防御性的）；m-di-dbl 真等价剔除。
```

**定向分解**（decomposition.py，逐行）：

```python
w = pretrained_w @ fix_covariance_matrix       # (out, in) @ (in, in)
U, S, V = torch.linalg.svd(w, full_matrices=False)
V = V.transpose(0, 1)                          # (in, rank)
V = (V.t() @ cov_inv).transpose(0, 1)          # 上下文恒等式的关键一步
```

**分量选取**（decomposition.py，逐行）：

```python
if not first_eigen:        # KPA：最后 r 个（最小奇异）
    U = U[:, -r:];  S = S[-r:];  V = V[:, -r:]
elif first_eigen:          # IPA：前 r 个（following PiSSA）
    U = U[:, :r];  S = S[:r];  V = V[:, :r]
weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0, 1)
```

**sigma_fuse 三态**（CorDA_adapter.__init__，逐行）：

```python
if sigma_fuse == 'UV':
    self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
    self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
elif sigma_fuse == 'U':
    self.ALinear.weight.data = U.mul(S).contiguous()
    self.BLinear.weight.data = V.t().contiguous()
elif sigma_fuse == 'V':
    self.ALinear.weight.data = U.contiguous()
    self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
```

**前向**（CorDA_adapter.forward，逐行）：

```python
y = self.BLinear(inp)          # (n -> r)
y = self.ALinear(y) + F.linear(inp, self.weight_residual)   # (r -> m) + 冻结残差旁路
```

## 判题域剔除与红线

- **剔除**：模型级 build_model 遍历/lm_head 跳过、act_aware 分支（scaling_diag_matrix/fisher_info）、NaN/inf 检查的 print 分支、CUDA empty_cache。
- **保留**：阻尼求逆循环、cov_aware 定向分解、KPA/IPA 选取、sigma_fuse 三态、冻结残差、前向。
- **锚定**：S/U/V fixtures 锚定（float32 管线）；全分解恒等式 \(U\,\mathrm{diag}(S)\,V^\top=W\)。

## 关键怪癖（回源逐行确认）

- **恒等式只对全分解成立**：\(r=rank\) 时协方差被逆精确消去；\(r<rank\) 的重构只是 rank-r 近似，余量进残差（沙箱先按 r=2 验证恒等式踩坑）。
- **\(V\) 的两次转置方向**：官方是 `(V.t() @ cov_inv).transpose(0,1)`——cov_inv 从**右侧**乘 \(V^\top\)；写成 `cov_inv @ V.t()` 是可抓变异。
- **fix 对称**：`fix = cov + cI` 对称 → `W @ fix.t()` 与 `W @ fix` 真等价（m-cd-mul 剔除）。
- **官方 SVD 管线 float32**：`pretrained_w.float()` 后进 svd；恒等式断言容差 1e-5（非 1e-10）。
- **KPA 取尾部、IPA 取头部**：`S_i[0] > S_k[-1]` 是模式区分断言；残差范数 KPA > IPA。
- **阻尼补偿用 mean(diag)**：不是 max(diag) 也不是 1e-6 常数（act_aware 分支才是 +1e-6）。

## 变异测试（17/17，2 等价剔除）

- m-di-dbl（damp 翻倍倍率）：判题域内循环恒首步收敛，倍率不可观测，真等价剔除。
- m-cd-mul（fix 转置）：fix = cov + cI 对称，真等价剔除。
- 17 个变异覆盖：damp 起点/补偿公式/加减号、w 乘法/full_matrices/V 两次转置方向/cov_inv 乘向/KPA-IPA 互换/残差加减号/r 校验×2、fuse 四态、前向 A-B 顺序/残差丢弃。

## 与已有题的划界

- **LoRA-GA（809）/LoRA-SB（808）**：SVD 源是**梯度/符号梯度**（数据无关），CorDA 源是**权重×协方差**（上下文相关）且 \(C^{-1}\) 乘回。
- **PiSSA/LoFTQ**：题库暂无 PiSSA 题；LoFTQ（量化初始化）走 NF/量化路径，无协方差账。
- **TinyLoRA（795）**：冻结 SVD 因子但源是权重 W 的直接 SVD，无上下文定向与 KPA/IPA。
- 题库无「协方差/逆矩阵」账（grep 已查重）。
