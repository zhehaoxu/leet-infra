# 解题笔记 — OSF

## 官方账

- HF PEFT `peft/tuners/osf/utils.py`（decompose_weight_matrix / project_gradient_to_orthogonal_space）+ 文档（effective_rank 语义、reconstruct）；持续学习 + 自适应 SVD。
- 判题域：SVD 切分、重构互逆、秩解析、单进程梯度投影。分布式 all-reduce 不在域内。

## 踩坑

- **测试副本名不能叫 `os.py`**：撞标准库 `import os`，`__import__('os')` 返回标准库模块（AttributeError: module 'os' has no attribute 'solve'）。
- **投影正交性需要正交因子**：U_high/V_high 用随机高斯时 `Uᵀ(proj)=0` 不成立（只有 UᵀU=I 才对）——必须用 `torch.linalg.qr(randn)[0]` 构造列正交基（V_high 取 `.T` 得行正交）。这是本树第一版两处正交性断言全挂的根因。
- **fp32 管线容差**：官方 decompose 强制 `to(torch.float32)` 再 cast 回，float64 test 下重构/最优近似恒等式差 ~1e-6，容差 1e-5（与 CorDA 教训一致）。
- **effective_rank 是保留秩**：可训练秩 = min_dim − preserved，与 LoRA r 相反；float 分数、None 默认 50%、出界 ValueError 三分支都要覆盖。
- **U/V 投影公式不对称**：U 左乘投影矩阵、V 右乘 Gram——照抄一边会错。

## mut 15 变异清单

decompose：截断头→尾、去 min 夹取、rank_high ±1、U_low 边界偏移；reconstruct：high−low、丢 diag（形状错）；resolve：None 0.5→0.25、int→round、分数边界含 0；project_u：直接返回原梯度、mm 转置错位（形状错）、减→加；project_v：直接返回、Gram 形状错（VVᵀ）、减→加。15/15 全 CAUGHT。
