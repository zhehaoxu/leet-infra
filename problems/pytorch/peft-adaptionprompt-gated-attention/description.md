# AdaptionPrompt 门控注意力分支

LLaMA-Adapter v2（论文 *LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention*, arXiv 2303.16199）在 HF PEFT 中以 `adaption_prompt` tuner 落地（`peft/tuners/adaption_prompt/layer.py` 的 `AdaptedAttention`）。它把可训练的「适配提示」注入注意力分支：在注意力的 key/value 侧插入 `adapter_len` 个可学习 token，通过一个**零初始化门控**把这些 token 的注意力输出残差加回原注意力输出——训练起点模型行为与原模型完全一致。

## 机制拆解

设 wrapped 注意力模块的原输出为 `attn_output`（bsz, q_len, hidden），适配提示参数为：

- `adaption_prompt`：形状 (1, adapter_len, hidden)，`normal_()` 标准正态初始化，**batch 维为 1（跨 batch 共享）**；
- `adaption_gate`：形状 (1,)，零初始化。

### 1. 适配 K/V 投影

把 prompt 过原模块的 k_proj / v_proj（两个独立投影）：

```
key   = prompt @ k_w.T   # (1, L, kv_heads*head_dim)
value = prompt @ v_w.T
```

再按 GQA 复制：view 成 `(1, L, kv_heads, head_dim)` 后 **先 transpose(1,2) 到 `(1, kv_heads, L, head_dim)`**，然后 `repeat_interleave(factor, dim=1)` 沿 KV 头维分组复制到 `(1, H, L, head_dim)`，使得查询头 h 读取 KV 组 `h // factor`。注意 `repeat_interleave` 产生 `[g0, g0, g1, g1]` 的相邻分组布局；`torch.repeat` 会交错成 `[g0, g1, g0, g1]`，是不同的广播语义。

### 2. 门控注意力分支

对已经过 rotary 的查询 `q`（bsz, H, q_len, head_dim）：

```
scores = q @ adapter_k^T / sqrt(head_dim)          # (bsz, H, q_len, L)
probs  = softmax(scores, dim=-1, dtype=float32)     # 显式 fp32 上转换
scores = gate * probs                               # 门控乘在 softmax 之后
adapter_output = (scores @ adapter_v).transpose(1,2).reshape(bsz, q_len, -1)
adapter_output = adapter_output @ o_w.T             # o_proj
```

三个判题敏感点：

- **门控在 softmax 之后**：`g·softmax(s)` 与 `softmax(g·s)` 不等价（官方是前者）；
- **softmax 显式 fp32 上转换**：即便输入是 float64，softmax 在 fp32 域计算再转回——数值上与纯 fp64 softmax 有差；
- **缩放在 softmax 之前**：`/sqrt(head_dim)` 作用于 scores 而非 probs。

### 3. 残差合并

```
output = attn_output + adapter_output
```

零初始化 gate 保证训练第 0 步 adapter_output 恒为 0，合并是恒等操作——这是 LLaMA-Adapter v2 「zero-init attention」的核心技巧。

### 4. 参数账

可训练参数 = `adapter_len × hidden + 1`（prompt + 标量门控）。对比逐层存 K/V 的完整前缀方案（`2 × adapter_len × hidden`），AdaptionPrompt 只存一份共享 prompt，省一半。

## 判题域

单入口 `solve(mode, **kw)`，五个 mode：

| mode | 输入 | 输出 |
|---|---|---|
| `init_params` | adapter_len, hidden, seed, dtype | `{"prompt": (1,L,H), "gate": (1,)}`（RNG 域显式 generator） |
| `adapter_kv` | prompt (1,L,H), k_w, v_w, num_heads, head_dim, factor | `{"adapter_k", "adapter_v"}` (1,H,L,hd) |
| `adapter_attn` | q (bsz,H,q_len,hd), adapter_k/v, gate, head_dim, o_w | `{"probs", "adapter_output"}` |
| `combine` | attn_output, adapter_output | 相加 |
| `param_count` | adapter_len, hidden | `{"trainable", "full_prefix", "saving"}` |

数值锚点用 float64 域；softmax 的 fp32 上转换是官方行为，必须保留。
