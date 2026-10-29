# 详解：Gemma2 解码层的整层装配

## 出处

`transformers==5.20.2` 的 `models/gemma2/modeling_gemma2.py`：

- `Gemma2RMSNorm.forward`：`output * (1.0 + self.weight.float())`，权重按 0 初始化；
- `Gemma2Attention.__init__`：`scaling = query_pre_attn_scalar**-0.5`，`sliding_window = config.sliding_window if layer_type == "sliding_attention" else None`；
- `Gemma2Attention.forward`：`self.attn_logit_softcapping` 一路传到 attention 接口；
- `eager_attention_forward`：先 `matmul * scaling`，再 `tanh(w/softcap)*softcap`，**最后**才 `attn_weights + attention_mask`；
- `Gemma2DecoderLayer.forward`：`residual + post_attention_layernorm(attn)`、`residual + post_feedforward_layernorm(mlp)` 两处 post-norm；
- `Gemma2MLP`：`down_proj(act_fn(gate_proj(x)) * up_proj(x))`，`hidden_activation="gelu_pytorch_tanh"`（GeGLU）；
- `masking_utils.sliding_window_overlay`：`kv_idx > q_idx - sliding_window`，与因果条件 `kv_idx <= q_idx` 取交。

本题是"整层装配"题：单点的滑窗掩码与 softcap 在 operators 合集里各有独立题（`sliding-window-attention` / `attention-softcap`），这里考的是它们与双 post-norm、\((1+w)\) 归一化组合后的**数据流**。

## 核心与易错点

1. **softcap 在掩码之前**，这不是风格问题而是数值正确性问题。掩码写入的是 \(-\infty\)；若先掩码再算 \(\text{softcap}\cdot\tanh(L/\text{softcap})\)，被掩位置得到 \(\text{softcap}\cdot\tanh(-\infty)=-\text{softcap}\)——一个**有限**值。softcap 越小，这个"假 logit"离 0 越近，softmax 给被掩 key 的权重越大。`check_softcap_before_mask` 用 `softcap=0.5` 放大这个差异，并同时跑"先掩码"的 oracle 变体，断言两者**必须不同**。
2. **两处 post-norm 作用在子层输出上**，而不是残差流上。写成 `x = RMSNorm(residual) + attn` 形状完全一致、不报错，但语义变成了"归一化残差流"，Gemma2 的 pre+post 混合结构就不存在了。oracle 的 `norm_target="residual"` 变体专门用来断言这种写法会被判错。
3. **\((1+w)\) 而不是 \(w\)**：Gemma2 把 gain 初始化成 0，让"零初始化 = 恒等"。若实现成 \(x\cdot w\)，初始 gain=0 会把整层输出清零。`check_norm_uses_one_plus_weight` 用 `post_attention` 的 gain=\([1,0]\)：正确实现给 5.242639，写成 \(w\) 会退化成 gain=0 的 3.828426。
4. **缩放取 `query_pre_attn_scalar**-0.5`**，不是 `head_dim**-0.5`。Gemma2-2B 里二者都是 256，所以这个 bug 在真实权重上完全不可见；本题的配置里 `query_pre_attn_scalar=8` 而 `head_dim=2`，一用 head_dim 就露馅。
5. **MLP 是 GeGLU 不是 SwiGLU**：`gelu(gate, approximate="tanh")` 与 `silu(gate)` 在 \(x>3\) 时都逼近 \(x\)，但在 1~2 区间差 10% 以上。用错激活示例行会立刻偏掉。
6. **滑窗含当前 token**：允许条件是 \(k>q-w\)，窗口宽度是 \(w\)（\(k=q\) 必可见）。写成 \(k\ge q-w\) 会让窗口变成 \(w+1\) 个 token，第 \(q-w\) 个 token 多出来。
7. **GQA 的复用是连续分组**：`repeat_kv` 用 `expand(...).reshape(...)`，第 \(h\) 个 query 头对应第 \(\lfloor h/r\rfloor\) 个 kv 头。写成 \(h\bmod r\) 或 `torch.cat` 平铺（头序变成 `[k0,k1,k0,k1]`）在 \(H_{kv}<H_q\) 时结果不同。
8. **softcap 的作用是压缩 logit 差距**：softmax 对整行加常数不变，所以 softcap 真正改变的是极值间距。示例里对角 logit 2.8284 被压到 2.7554，权重从 0.9442 降到 0.9402；当 logits 远大于 softcap 时，权重趋于均匀。
9. **位置编码在这里只是"已算好的 cos/sin"**：RoPE 在 \((B,H,S,d_h)\) 上按 head 广播（`unsqueeze(1)`），与本层其他部分解耦，避免和 `llm-rope-apply` 重复。
10. **精度**：参考解在 float64 下算完整层再转 float32；HF 在 float32 下算，故与 HF 对拍的阈值取 `2e-5`，而与 numpy oracle（同为 float64）取 `1e-5`。

## 对拍口径

判题有三层：

- **示例值**：单位阵 + 全零 gain 的可手算配置，`output` 四行与 `attn_weights[0,0]` 四行都六位小数硬编码；
- **结构断言**：① 滑窗支持集（`window=3`、`window=1`、`full_attention` 三种）——可见位置权重严格为正、被掩位置严格为 0、每 `(batch, head)` 行和为 1；② softcap 必须先于掩码（与 `mask_first` oracle 变体对照且必须不同）；③ \((1+w)\) 归一化（两个手算值 + 二者必须不同）；④ post-norm 必须作用于子层输出（与 `norm_target="residual"` 变体对照）；⑤ 缩放必须用 `query_pre_attn_scalar`（与按 `head_dim` 缩放的 oracle 对照）；⑥ GQA 头映射（\(H_q=4,H_{kv}=2\) 时输出与权重同时对上）；
- **双 oracle**：① 独立 numpy `_oracle`——RMSNorm、掩码、softmax、GeGLU 全部用显式公式写，投影用 `@` 实现，覆盖 5 组形状 × 层型 × softcap 配置 + 两组 eps，输出与 `attn_weights` 都逐元素对拍；② `transformers` 的 `Gemma2DecoderLayer`（eager、同权重、同一份 4D 加法掩码与 `position_embeddings`）端到端对拍 4 组配置，阈值 `2e-5`。

另有 20 组非法输入必须以 `ValueError` 拒绝：非 3 维 `hidden_states`/`cos`、`cos`/`sin` 形状不一致、gain 长度不对、q/k/o/down 的行列数与 `head_dim` 不整除、未知 `layer_type`、`sliding_window` 为 0 或浮点、`query_pre_attn_scalar` 非正或 NaN、`softcap` 非正、`rms_eps` 非正或无穷、`head_dim` 为奇数。

## 复杂度

注意力 \(O(B\cdot H_q\cdot S^2\cdot d_h)\)（GQA 下 kv 侧只存 \(H_{kv}\) 份），投影与 MLP 共 \(O\big(BS(d\cdot H_qd_h + d\cdot H_{kv}d_h + 2dI)\big)\)；空间上除输出与权重外只需保留当前隐状态，\(O(BSd + B H_q S^2)\)。
