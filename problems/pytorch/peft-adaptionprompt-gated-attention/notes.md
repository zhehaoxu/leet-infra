# 解题笔记 — AdaptionPrompt

## 官方账

- HF PEFT `peft/tuners/adaption_prompt/layer.py`（`AdaptedAttention`，Llama 分支）；论文 LLaMA-Adapter v2（arXiv 2303.16199），官方 repo ZrrSkywalker/LLaMA-Adapter。
- 判题域取纯函数部分：prompt/gate 初始化、GQA K/V 复制、门控注意力分支、残差合并、参数账。transformers 依赖（rotary 重算 q、mask）不在域内——q 由用例直接给定。

## 踩坑

- **transpose 顺序**：官方是 `view(1, L, kv, hd)` → `transpose(1, 2)` → `repeat_interleave(factor, dim=1)`。若漏 transpose 先 repeat_interleave，复制的维度变成 L 维，形状直接错（(1, 6, 2, 2)）。
- **repeat vs repeat_interleave**：dim=1 是 KV 头维，`repeat_interleave` 产生 `[g0,g0,g1,g1]`（相邻分组），`repeat` 交错成 `[g0,g1,g0,g1]`。组恒等断言应为相邻头相同（ak[0,0]==ak[0,1]），不是 h 与 h+factor。
- **gate 乘在 softmax 之后**：`softmax(g·s)` 是 pre-gate，与官方 `g·softmax(s)` 数值不同，好变异点。
- **softmax fp32 上转换**：`dtype=torch.float32` 即使在 float64 全局下也有效——fp64 计算的 softmax 与 fp32 域 softmax 转回 fp64 有差，test 必须锚定上转换行为。
- **test 顶部必须 `set_default_dtype(float64)`**：否则默认 float32 流，锚值全错（本树第一版 anchor 全挂的根因）。
- **kv==hd 时 view 转置真等价**：shape 家族必须含 kv≠hd 用例（kv=1, hd=4），否则 (kv,hd)→(hd,kv) 的 view 变异幸存。
- **av 值断言不能省**：只对 ak 对拍时，v_w→k_w、av view 转置两类变异全部幸存。
- mut anchor 撞上下文：`gate = torch.zeros(1, dtype=dtype)` 唯一；`return attn_output + adapter_output` 在 combine 里出现两次（m-combine-sub/m-combine-drop 同 anchor 需错开）——实际用同一 anchor 两次 replace(…, 1) 时第二次替换的是已变异文本的父串，需保证 anchor 唯一。最终 combine 两变异 anchor 相同但按序替换互不干扰（第一个变异替换后第二个 anchor 已不在），按 dict 顺序各自独立 tempdir 跑，无冲突。

## mut 17 变异清单

init：normal std / gate ones；kv：漏 transpose / repeat 交错 / v_w→k_w / view 转置；attn：scale /√hd→/hd / fp32 上转换 / pre-gate / o_proj 系数 / softmax dim / 漏 transpose(1,2)；combine：相加→相减 / 丢弃分支；param_count：full_prefix 系数 / saving 符号 / trainable 常数。17/17 全 CAUGHT。
