# 解题笔记

## 思路

官方 SVDLLM.py 的 per-Linear 压缩管线，取两段确定性核心：

1. **whitening**（L187-284）：白化域 SVD + 参数预算截断 + √Σ 因子拆分（v 侧 L⁻¹ 补偿）；
2. **local_update / fasterprune**（L442-498）：lstsq 闭式更新 + append 因子 + 相对误差。

上游不入判题域：校准 H 累积（Σinpᵀinp hook）、Cholesky 产出 L（profiling 阶段）、direct_update=False 时 L 的非正定修正（eigenvalue 分支在 profiling 侧；本题只保留 inv 失败的 +1e-6·I 防御）。

与现有题划界：
- **771 LoSparse**：W 自身谱 + √diag 拆分 + 残差结构化剪枝，无白化无 L⁻¹；
- **520 SparseGPT**：逐列 OB 剪枝 + 误差传播，非整体低秩；
- **770 SliceGPT**：正交旋转切边（不变秩），SVD-LLM 是降秩逼近。

## 关键实现点

- **k 上界是死代码**：`ratio ≤ 1` 时 `k = int(MN/(M+N)) ≤ min(M,N)/2 < min(M,N)` 恒成立（MN/(M+N) 是两组调和平均的一半），只查 `k < 1`——与 771 的「正比例 ceil 恒 ≥1」同一教训的反向版。
- **√Σ 拆分的数学**：`svd_u @ svd_v = U_k Σ_k V_k L⁻¹`；`svd_u @ (svd_v @ L) = U_k Σ_k V_k` 恰是 `W@L` 的 k 秩最优逼近（Eckart–Young）——测试用相对误差 < 0.5 断言（k=3、谱衰减慢时绝对误差 2.28 是正常的）。
- **updated_output 官方口径**：`x @ truc_vᵀ @ Σ @ updated_uᵀ`，与 `append_u @ append_v` 组合（√Σ 夹心）在代数上不等价（Σ 对角但 √Σ@A@√Σ ≠ Σ@A）——判题按官方分别复刻，`updated_error` 用 append 组合的输出定义。
- **L 的 fixture**：`H = raw@rawᵀ + 0.5·I` 的 Cholesky——保证正定可逆；奇异用例把 L 某行清零触发 `+1e-6·I` 防御（inv 失败分支），结果必须有限且与 oracle 的 `np.linalg.inv` 防御一致。
- **direct_update 分支**：`scaling_diag_matrix=None` 且不许传 L（传了报错）——官方 `direct_update=True` 走 `svd(W)` 且 v 侧无 L⁻¹。

## 测试设计（8 例）

- 白化/direct 双路径全对拍（k、因子、重构、updated_uᵀ、append 因子、两个误差）；
- L⁻¹ 可观测：svd_v ≠ 朴素 √Σ@Vᵀ（白化丢失变异的判据）+ `svd_u@(svd_v@L) ≈ W@L` 相对误差有界；
- 闭式补偿不劣化：`updated_error ≤ error`（lstsq 最小二乘性质）；k 增大误差单调不增；
- 奇异 L 防御：结果有限、与 oracle 一致；
- k 公式：ratio 四点单调 + ratio=1.0 合法 + ratio 过小报 `k (0) 必须 >= 1`；
- 小范数 y（×0.005，x 同步缩放保持相容）：抓误差分母 `max(norm,1)` 类变异；
- 纯函数 + 校验（形状/整型张量/direct+L 组合/ratio 边界与 bool）。

## 变异收口

39 变异全 CAUGHT（无等价剔除）。覆盖：k 公式四变异 + 下界校验删除 + 消息；白化丢失/转置/direct 翻转；截断三侧尾段反转 + k±1；√Σ 三变异（u 侧 Σ、v 侧 Σ、sqrt 丢弃）；L⁻¹ 丢弃/换 L/防御 ε 变异/防御删除；闭式更新四变异（漏 Σ、Vᵀ 不转置、lstsq 参数交换、updated_uᵀ 用 stale truc_u）；append_v/new_w 的 sqrt 变异；误差分母/分子/updated 用 stale 重构；校验七变异。

## 工程学备注

- 测试矩阵维度必须让 `k = int(MN·ratio/(M+N)) ≥ 1`：12×8@0.2 → k=0 全线报错；放大到 40×24（k=3@0.2）。
- y 缩放破坏 x/y 相容性会让 lstsq 误差爆炸——缩放必须 x/y 同步（`x = x0·s`、`y = (x0@Wᵀ + noise)·s`）。
- f-string 消息断言别硬编码维度字符串（改维度后 `(8, 8)` 全部漏改，args_validation 假失败两轮）。
