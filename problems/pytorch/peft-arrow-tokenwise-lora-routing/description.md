# PEFT：Arrow LoRA 库 token 级路由

## 题目描述

**Arrow**（arXiv:2405.11157，NeurIPS 2024，已合入 HF PEFT `ArrowConfig`/`create_arrow_model`，PR #2644）用**免梯度、token 级的 MoE 路由**复用预训练 LoRA 适配器库：每个适配器贡献一个**原型**（\(\Delta W = B@A\) 的 top 右奇异向量，用**幂迭代** \(v \mapsto A^\top (B^\top B)(Av)\) 求得，不物化 \(\Delta W\)）；每个 token 与原型做**符号无关**的相似度比较，路由到 top-k 个最相关适配器，softmax 温度归一后线性组合专家输出。

本题复刻官方 PEFT 实现（`arrow.py` 的 `ArrowLoraLinearLayer` + `check_loaded_lora_compatibility_arrow`）的确定性账：

- **幂迭代原型**：v₀ = randn(in, `rng_seed` 锚定) 除以范数（eps=1e-8）；迭代 `iters=15` 步 `w = Aᵀ(BᵀB)(Av)`、`v = w/(|w|+eps)`；全程 float32（官方数值稳定性），返回 fp32。
- **原型库**：每适配器一个向量；官方 `build_prototypes` 把 fp32 结果 cast 回 \(A\) 的 dtype 后 stack 成 \((E, in)\)。
- **路由系数**：`sim = |tok @ protosᵀ|`（**绝对值**——sign-agnostic）；逐 token top-k；未选中位置填 \(-\infty\) 后 `softmax(score / temperature)`——**非 top-k 系数严格为 0**，选中系数和为 1。
- **专家输出与组合**：\(z = \) einsum("tf,erf→ter")、\(y = \) einsum("ter,eor→teo")；乘**逐适配器** scaling（task_adapter_names 顺序）；`delta = einsum("te,teo→to", coeff, y)` 后 reshape 回 (B, *rest, out)。
- **库兼容校验**：所有适配器必须共享 rank r、A/B 形状一致、target_modules 集合一致，否则 ValueError。

## 任务

- `mode="singular_vec"`：a、b（(r,in) / (out,r)）、iters、seed、eps 入参，返回 fp32 单位向量。
- `mode="prototypes"`：pairs（(A,B) 列表）、iters、seed 入参，返回 (E, in) 原型库（dtype 跟随 A）。
- `mode="route_coeffs"`：tok、prototypes、top_k、temperature 入参，返回 (coeff, idx)；非选中系数恰为 0。
- `mode="expert_outputs"`：tok、a_stack、b_stack、scales 入参，返回 (t, E, out) 带缩放专家输出。
- `mode="arrow_delta"`：x（任意前导维）、prototypes、a_stack、b_stack、scales、top_k、temperature 入参，返回 x 形状末维换 out 的 delta。
- `mode="check_compat"`：ranks、shapes、module_sets 入参，返回 (sorted modules, r)；不一致抛 ValueError。

## 示例

seed 0/1/2/4，rng_seed=42，float64 管线（幂迭代内部 fp32）：

```
singular_vec(A (3,8), B (6,3), seed=42) -> v (8,), v[0] = 0.6975467205047607
prototypes(pairs×3, seed=42) -> (3,8), protos[2,3] = -0.170495867729187
route_coeffs(tok (5,8), protos (4,8), top_k=2, T=1) -> coeff (5,4)
  coeff[0, idx[0,0]] = 0.7688820194964578；非选中 = 0；T→0 趋近 one-hot
arrow_delta(x (2,3,8), ...) -> (2,3,6), delta[0,0] = 2.7192971987121317
check_compat([4,4], 同形状, [{"q","v"},{"v","q"}]) -> (["q","v"], 4)
```

## 约束

- 相似度取**绝对值**（sign-agnostic）；奇异向量本身 ± 不定，下游必须符号无关。
- 幂迭代迭代次数不足会显著偏离主特征方向（iters=2 与 15 差 ~0.26）。
- 未选中的路由系数**严格为 0**（-inf → softmax 下溢），不是小数。
- temperature 除在 softmax **之前**；T→0 退化为 argmax one-hot。
- scaling 按适配器作用在专家输出上，进入加权和之前；不同 top_k/温度改变路由。
- 原型库 dtype 跟随 A（fp32 迭代结果 cast 回来），路由全程无额外 dtype 转换。
