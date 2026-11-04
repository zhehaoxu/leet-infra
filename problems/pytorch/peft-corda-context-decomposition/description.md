# PEFT：CorDA 上下文定向分解适配

## 题目描述

**CorDA**（arXiv:2406.05223，iboing/CorDA）把下游任务的**上下文**注入 LoRA 初始化：用少量校准样本收集输入激活的**协方差矩阵** \(C\)，对 \(W\cdot C\) 做 SVD 定向分解，再把 \(C^{-1}\) 乘回 \(V\)——分解方向由任务数据决定。知识保留模式（KPA）取**最小**的 \(r\) 个奇异分量做可训练适配器（世界知识留在冻结残差里缓解遗忘）；指令预览模式（IPA，沿用 PiSSA）取**最大**的 \(r\) 个。

本题复刻官方实现（`cordalib/decomposition.py`）的确定性账：

- **阻尼求逆**：\(fix = C + I\cdot\mathrm{mean}(\mathrm{diag}(C))\cdot damp\)，`damp` 从 0.01 起、\(\|fix\cdot fix^{-1}-I\|\ge 0.05\) 时翻倍重试。
- **定向分解**：\(w = W\cdot fix\)，`linalg.svd(w)` 后 \(V \leftarrow (V^\top C^{-1})^\top\)。**上下文恒等式**：全分解时 \(U\,\mathrm{diag}(S)\,V^\top = W\) 精确成立（协方差被逆精确消去）；\(r<rank\) 时余量进入冻结残差。
- **分量选取**：`first_eigen=False`（KPA）取 \(U[:,-r:],\,S[-r:],\,V[:,-r:]\)；`first_eigen=True`（IPA）取前 \(r\) 个。
- **冻结残差**：\(R = W - U\,\mathrm{diag}(S)\,V^\top\)。
- **sigma_fuse 三态**（均满足 \(A@B = U\,\mathrm{diag}(S)\,V^\top\)）：`UV`（\(A=U\sqrt S\)、\(B=V^\top\sqrt S\) 按列缩放）；`U`（\(A=US\)、\(B=V^\top\)）；`V`（\(A=U\)、\(B=V^\top S\)）。
- **前向**：\(y = A\,(B\,x) + R^\top x\)；初始化处 \(y = W^\top x\) 精确成立（零增量性质）。

## 任务

- `mode="damped_inv"`：cov、damp（默认 0.01）、tol（默认 0.05）入参，返回 (fix, cov_inv, damp)。
- `mode="decompose"`：weight、cov、r、first_eigen 入参，返回 (U, S, V, residual, fix, cov_inv, damp)；r<=0 或 r>rank 抛 ValueError。
- `mode="fuse"`：U、S、V、fuse 入参，返回 (A, B)；非法 fuse 抛 ValueError。
- `mode="forward"`：x、a_w、b_w、residual 入参，返回 y。
- SVD 管线按官方 float32（`.float()`），阻尼求逆随输入 dtype。

## 示例

\(d_\text{out}=6, d_\text{in}=5, r=2\)，W = randn(6,5)、X = randn(64,5)（seed 2），cov = \(X^\top X/64\)：

```
damped_inv(cov) -> damp = 0.01（良态协方差一步收敛）
decompose(W, cov, 2) -> S = [1.2007561922073364, 0.7124348878860474]（float32 管线锚定）
  U[0,0] = 0.3145820200443268, V[0,0] = -0.2790605425834656
decompose(W, cov, 5)（全分解）-> U diag(S) V^T == W（相对误差 < 1e-5，float32 SVD 精度）
fuse(U, S, V, "UV") -> A=U*sqrt(S), B=V^T*sqrt(S)；A@B == U diag(S) V^T
forward(x, A, B, R) == x @ W^T（零增量）
```

## 约束

- `2 <= r <= min(in, out)`；r 非法抛 ValueError。
- fix = cov + I·mean(diag)·damp 中的 damp 是返回值的一部分（翻倍循环的起点可入参）。
- KPA/IPA 两种选取互斥且都必须与冻结残差一起精确重构 W。
