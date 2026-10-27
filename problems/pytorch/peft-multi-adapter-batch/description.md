# PEFT：多 Adapter 批量热切换路由

## 题目描述

多 LoRA 服务（S-LoRA, OSDI 2024, arXiv:2311.03285；punica, arXiv:2310.18547）的核心原语：同一个冻结基座，一个 batch 里不同请求各挂**不同的 adapter**，一次前向全算完——不为每个 adapter 复制模型，也不按请求串行。批内路由：请求 \(i\) 用 adapter \(k_i\)，基座输出对所有人共享、低秩增量按行取各自的因子与缩放：

\[y_i=W_0x_i+s_{k_i}\,B_{k_i}A_{k_i}x_i\]

（\(A_k\in\mathbb R^{r\times d_\text{in}}\) 下投影、\(B_k\in\mathbb R^{d_\text{out}\times r}\) 上投影、\(s_k\) 为该 adapter 的缩放——合并/加权多个 adapter 时 \(s\ne1\)。）系统层把对应的批处理 kernel 叫 SGMV（segmented gather-matrix-vector）：batch 内按 adapter 分段做 GEMV。

## 任务

实现按 adapter 路由的批量前向。

- 输入：\(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)；\(A\in\mathbb R^{K\times r\times d_\text{in}}\)、\(B\in\mathbb R^{K\times d_\text{out}\times r}\)（\(K\) 个 adapter）；\(\texttt{scales}\in\mathbb R^{K}\)；\(\texttt{ids}\in\mathbb Z^{N}\)（每样本的 adapter 索引，\(0\le k<K\)）；\(x\in\mathbb R^{N\times d_\text{in}}\)。
- 输出 \(y\in\mathbb R^{N\times d_\text{out}}\)（float32）。

**约定**：内部用 float64 计算、输出 float32；不使用任何现成 PEFT/vLLM 库；只算前向合成（显存与调度账归合集内 serving 题）。容差 \(10^{-5}\)。

## 提示

- 基座部分与 adapter 无关：先算 \(xW_0^\top\)，再按 id 分组累加增量。
- 全零缩放时应精确退化为 \(xW_0^\top\)（基座共享语义）。
- ids 全相同时应与单 adapter 的 \(W_0x+s\,BAx\) 逐位一致。
