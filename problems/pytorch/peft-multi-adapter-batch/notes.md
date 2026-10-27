# PEFT：多 Adapter 批量热切换路由——算法详解

## 1. 出处与系统背景

S-LoRA（Sheng et al., OSDI 2024, arXiv:2311.03285）与 punica（Chen et al., arXiv:2310.18547）确立了"一基座千 adapter"的服务形态：LoRA 权重只占基座参数的 1% 以下，显存里可同时驻留几千个 adapter、按请求动态挂载。decode 阶段的瓶颈是"batch 内不同请求不同 adapter"的 GEMV——punica 用自定义 **SGMV** kernel 把按 id 分段的 gather-matvec 融合，S-LoRA 用 unified paging 管理显存。本合集 #79⑦ 已考显存/LRU 账，本题只考**前向合成**。

## 2. 前向的分解（判题考点）

\[y=\underbrace{xW_0^{\!\top}}_{\text{共享基座}}+\underbrace{\sum_k \mathbb 1[\text{ids}=k]\,s_k\,(xA_k^{\!\top})B_k^{\!\top}}_{\text{按行路由的增量}}\]

实现等价于"按 k 分组、组内批量 GEMV、按行 scatter 回"，判题钉四个语义：

- `check_zero_scales_identity`：所有 \(s_k=0\) → 精确等于 \(xW_0^\top\)（增量全灭、基座共享）。
- `check_single_adapter_consistency`：ids 全为 0 → \(y=xW_0^\top+s_0(xA_0^\top)B_0^\top\)，与单 adapter 前向逐位一致（路由退化）。
- `check_scale_doubling_scope`：\(s_1\to 2s_1\) 后**只有 ids==1 的行**变化——路由按行隔离，任何串扰都是 bug。
- 缩放 \(s_k\) 是合并/加权多 adapter 的接口（如 adapter 混合），不是装饰——example 与随机对拍里都让它非 1。

## 3. 与本合集其他题的分工

- `peft-lora-forward`：单 adapter 前向；`peft-adalora-budget`：训练期秩分配。
- serving 显存/换入换出（LRU、冷启动）归 #79⑦。
- 本题：批内路由 + 缩放合成的**前向语义**。

## 4. 验证方法

`check_example` 手算（\(r=1,K=2,N=3\)、\(W_0\) 为单位阵）：样本 0/2 挂 adapter 0（增量 \(2\cdot[1,0]^\top\) 的缩放外积）、样本 1 挂 adapter 1 → \(y=[[3,0],[0,4],[3,1]]\)；`check_matches_oracle` 与逐样本逐标量重算对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
