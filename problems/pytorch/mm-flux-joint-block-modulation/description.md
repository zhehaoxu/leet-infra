# FLUX double block 调制账：AdaLN-Zero 六参数与门控残差

## 题目描述

FLUX 的 transformer 由 double block（图像流与文本流各自一套 AdaLN-Zero 调制 + joint attention）与 single block（两流拼接后共享一套调制）组成。block 的骨架是一个精确的调制-门控-残差协议：`silu(temb)` 过各自 Linear 生成 6 参数，调制 LayerNorm 输出、门控 attention 与 FF 分支、加回**原输入**。attention 内部（QKV+softmax+RoPE）已有别的题覆盖，本题把 attention 输出作显式入参，专考这条协议账。

## 规则

实现 `solve(hidden_states, encoder_hidden_states, temb, weights, attn_img, attn_txt)`（batch=1），按 diffusers `FluxTransformerBlock.forward` 逐句转写两条流（各一套 `img_*` / `txt_*` 权重），返回：

| 字段 | 形状 | 含义 |
|---|---|---|
| `hidden_states` | `[1, Li, D]` | 图像流块输出 |
| `encoder_hidden_states` | `[1, Lt, D]` | 文本流块输出 |
| `img_mod` / `txt_mod` | `[6, D]` | 六参数，chunk 顺序 `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]` |
| `img_norm_input` / `txt_norm_input` | `[Li, D]` / `[Lt, D]` | 调制后送入 attention 的值 |

每条流的五步（上游逐句）：

1. `emb = Linear(silu(temb))`，`chunk(6)` 成六参数
2. `x_mod = LayerNorm(x, eps=1e-6, 无仿射) * (1 + scale_msa) + shift_msa`（送 attention）
3. `x = x + gate_msa * attn`（**残差加回原输入，不是调制值**）
4. `h = LayerNorm(x) * (1 + scale_mlp) + shift_mlp`
5. `x = x + gate_mlp * ff2(GELU_tanh(ff1(h)))`

校验：batch ≠ 1 或非张量输入抛 `ValueError`。

## 示例

```
gate_msa = gate_mlp = 0（权重与偏置对应行清零）→ 块输出 = 输入（残差恒等）
six 参数 chunk 顺序错一行 → img_mod 账与输出双双偏离
```

## 约束

- `D=4` 级小规模，float64 判题；oracle 为纯 Python 逐元素手写（SiLU/GELU-tanh/LayerNorm/Linear）
- eps 的可观察面用近常数行（方差 ~1e-6）构造
